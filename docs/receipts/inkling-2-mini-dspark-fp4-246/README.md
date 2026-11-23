# Inkling 2 Mini DSpark draft head at BF16, FP8 and FP4, M5 Ultra: card 246's receipt

Apple M5 Ultra, 256 GB, macOS 27.0; the official `Inkling-2-Mini` checkpoint with its own `mtp.safetensors` as the
DSpark drafter; `tensorfold-native` from `zig build native -Doptimize=ReleaseFast` at d14f7c387. `INKLING2_DRAFT_HEAD`
picks the head (`fp4` when unset). Every run held `/tmp/tensorfold-model.lock` inside the Desk's Mac GPU window, one
at a time.

## What changed

DSpark's two full-vocabulary products, the unembed base (201,024 x 2,304) and the folded Markov Phi (201,024 x 256),
read a private copy packed at load: E2M1 (FP4) or E4M3 (FP8) codes with one BF16 scale per 32 columns, so FP4 is
4.5 bits a weight. Bytes a seven-row hold reads drop from 926 + 103 MB to 261 + 29 MB. The target's unembed and embed
are hashed before and after the drafter loads in the device test and stay byte-identical. Verify and committed state
are untouched: drafts only propose, and the target still verifies every one of them.

## Exactness

- `test-inkling2-metal-host` with `TF_INKLING2_METAL_DIR` and `TF_INKLING2_DSPARK_DIR` set, FP4 head: 88 passed,
  1 skipped (the float64 gate, skipped in Debug). This includes `Mini Metal opt-in DSpark drafting preserves target
  tokens and rolls back exactly`, which runs the drafted path against a draft-disabled reference token by token.
- `the packed kernels match their codes decoded on the host` checks both GPU products against a float64 decode of the
  packed codes. Swapping the FP4 nibble order in the shader fails it.
- In every bench cell below, the drafted greedy replies equal the draft-disabled ones (`serial`: 1 of 1), and every
  concurrent reply equals its run alone (3 of 3 at one stream, 12 of 12 at four), with 0 failed requests.

## Speed and acceptance, 1 and 4 streams

`bench.sh` as in card 210 (`--levels 1,4 --reps 3 --tokens 256 --temperatures 0 --alone --serial`), one session per
head, back to back:

| head | tok/s 1 stream (code / chat) | tok/s 4 streams (code / chat) | tokens a round, 1 stream (code / chat) | accepted drafts (code / chat) |
|---|---|---|---|---|
| BF16 | 116.4 / 81.1 | 242.3 / 171.1 | 3.459 / 2.485 | 180 / 149 |
| FP8 | 115.6 / 91.7 | 200.3 / 181.2 | 3.122 / 2.370 | 177 / 149 |
| FP4 | 109.3 / 93.2 | **276.5 / 215.0** | 3.241 / 2.349 | 180 / 146 |

- Acceptance: FP4 accepts the same 180 drafts as BF16 on code and 146 against 149 on chat (2.0% fewer). FP8 is 177
  and 149. Both are under the 5% bar.
- Rounds: FP4's one-stream round is 28.3 ms on code and 27.0 ms on chat, against BF16's 30.0 and 28.0. At four
  streams it is 38.4 / 37.5 ms against 41.4 / 41.0. Each of the four streams reads the copy, so the saving grows with
  streams.
- Each head ran one session of three reps. The one-stream rates move with the median rep's round count more than
  with the round time: FP4's 3.241 tokens a round on code against BF16's 3.459 is the same 180 accepted drafts
  spread differently across reps. A one-stream speed-up is not claimed from these runs.

## 64 streams

`--levels 64 --reps 1`, `--context 8192 --parallel 64`, no `--alone`/`--serial`:

| | code | chat |
|---|---|---|
| plain (no drafter) | 390.0 | 348.8 |
| BF16 head | 439.7 | 418.9 |
| FP4 head | 357.7 | 400.9 |

At 64 streams DSpark rounds carry 1.0-1.02 tokens. The drafts are mostly not attempted at this width, so these
single-rep rates are noise around the plain rate and say nothing about the head's precision.

## Against the fastest other engine

No other engine that serves Inkling 2 Mini is installed on this Mac: there is no `mlx_lm` (#4618), `llama-server` or
Strata. The bar is #4618's published 123 tok/s at one stream and 281 at four, as in card 210's receipt. Against it:

| | ours, best head | #4618 |
|---|---|---|
| 1 stream (code) | 116.4 (BF16) | 123 |
| 4 streams (code) | 276.5 (FP4) | 281 |
| 64 streams | 439.7 (BF16) | not published |

Not beaten: we are 5.4% under at one stream and 1.6% under at four. The 64-stream comparison and a same-machine run
of #4618 need that engine installed here. The Spark row waits for card 236's draft head on feat/211.

## Files

`bench.sh` and `bench64.sh` are card 210's `bench.sh`, with `--reps 3`, and with `--levels 64 --reps 1 --context 8192
--parallel 64`. `cells.jsonl` holds every cell's figures.
