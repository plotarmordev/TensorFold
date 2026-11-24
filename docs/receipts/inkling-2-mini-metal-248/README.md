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
streams (`bench.sh`). **Awaiting the Desk's Mac GPU window**; the figures go here when it runs, with the round ms
delta and the two submissions it no longer makes beside the probe's per-submission cost.

## Exactness

```
TF_INKLING2_METAL_DIR=$HOME/models/Inkling-2-Mini TF_INKLING2_DSPARK_DIR=$HOME/models/Inkling-2-Mini zig build test-inkling2-metal-host
TF_INKLING2_METAL_DIR=$HOME/models/Inkling-2-Mini TF_INKLING2_DSPARK_DIR=$HOME/models/Inkling-2-Mini zig build test-inkling2-metal-float64
```

These device checks require the Mac GPU window and both checkpoint variables; without them, the tests that need
weights skip. Greedy draft-vs-disabled parity, the 0/32 float64 comparison and 0 failed device requests remain
unverified; `zig build test test-golden` covers the host-only suites.

## Files

`run.sh`: the window sequence; `bench.sh`: one three-rep session from a given tree; `submission.swift`: one
submission's commit-and-wait cost.
