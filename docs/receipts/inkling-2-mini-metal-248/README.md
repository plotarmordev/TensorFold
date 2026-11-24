# Inkling 2 Mini on Metal, M5 Ultra: card 248's receipt

Apple M5 Ultra, 256 GB, macOS 27.0, the official checkpoint with its own `mtp.safetensors` as the DSpark drafter,
`tensorfold-native` from `zig build native -Doptimize=ReleaseFast`, `--context 65536`, greedy. Control: the base
branch's tip `90fac7967`. Candidate: this branch's head.

## What a round submits

A DSpark round is verify (the target's rows), keep (the staged taps' blits), absorb (the accepted rows into the
drafter's lane) and hold (the drafts). Control submits each of the four in its own command buffer and waits for
each. Candidate stages the keep on the host, then encodes its blits, the absorb and the greedy hold into one
command buffer at the next draft readback: **one blit and one serial compute pass, one commit and one wait**.
Verify still has its own forward submissions because its sampled tokens and kept paths are host inputs to the
following round. The three post-verify submissions become one.

## Measurement

`run.sh` under one lock: build phase (control and candidate, logged), one submission's host cost (`submission.swift`:
commit an empty command buffer and wait), then measure phase control/candidate/control, medians of 3 at 1 and 4
streams (`bench.sh`). Window of 25 Nov 01:00 HKT; machine load average 24–33 throughout (other builds running).
Control `90fac7967`, candidate `8b3ad49a3`, `bench_concurrent.py --tokens 256 --temperatures 0`, aggregate tok/s,
median of 3 (reps in brackets), 0 failed requests in every cell:

| cell | ctrl1 | cand | ctrl2 |
|---|---|---|---|
| code, 1 stream | 118.5 (118.5, 100.6, 122.9) | 113.4 (113.4, 109.2, 118.2) | 120.9 (120.6, 120.9, 121.1) |
| code, 4 streams | 298.4 (321.1, 298.4, 296.9) | 307.3 (309.2, 307.3, 304.1) | 323.8 (323.1, 323.8, 325.1) |
| chat, 1 stream | 85.6 (74.2, 88.4, 85.6) | 89.7 (89.7, 92.2, 83.0) | 94.9 (94.9, 95.6, 72.8) |
| chat, 4 streams | 264.6 (264.3, 264.6, 264.7) | 229.0 (229.0, 255.3, 208.8) | 175.8 (173.7, 175.8, 176.9) |

**Bar not reached**: 113.4 at 1 stream against #4618's 123, under both controls. Round gap p50 at 1 stream (code):
27.6 / 27.5 / 28.3 ms. One submission's commit-and-wait is 14.2 µs median (p95 44.8, min 9.9), so the two
submissions a round no longer makes are worth about 0.03–0.09 ms of a ~27.5 ms round, under 0.3%: below this
session's noise, and the controls' own drift (ctrl2 chat at 4 streams fell 34% from ctrl1) is far larger. A
round's time is GPU work, not submission overhead; the remaining lever is the verify and hold kernels themselves.

## Exactness

```
TF_INKLING2_METAL_DIR=$HOME/models/Inkling-2-Mini TF_INKLING2_DSPARK_DIR=$HOME/models/Inkling-2-Mini zig build test-inkling2-metal-host
TF_INKLING2_METAL_DIR=$HOME/models/Inkling-2-Mini TF_INKLING2_DSPARK_DIR=$HOME/models/Inkling-2-Mini zig build test-inkling2-metal-float64
```

`test-inkling2-metal-float64` passes on this branch's source (`8b3ad49a3`'s `dspark.zig`): 0 of 32 greedy tokens
differ from the float64 host reference and the planted control disagrees (21 minutes, mostly CPU under load 30).
The DSpark device tests (drafts equal to no drafts, the sixteen-stream chain, four-stream leave and join) are in
the Debug host suite (about 6.5 minutes on the GPU) and await the next Mac GPU window; the first window's attempt
deadlocked on `/tmp/tensorfold-model.lock` because the tests take it themselves, so run them without `lockf`.
The bench's 0 failed requests in all 36 runs hold; `zig build test test-golden` passes.

## Files

`run.sh`: the window sequence; `bench.sh`: one three-rep session from a given tree; `submission.swift`: one
submission's commit-and-wait cost; `profile.md`: where a one-stream round's 27 ms goes and the next lever.
