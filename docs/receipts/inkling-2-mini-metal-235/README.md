# Inkling 2 Mini on Metal, M5 Ultra: card 235's receipt

Apple M5 Ultra, 256 GB, macOS 27.0; the official checkpoint (`Inkling-2-Mini`, its own `mtp.safetensors` as the
DSpark drafter); `tensorfold-native` from `zig build native -Doptimize=ReleaseFast`, `--context 65536`. Control is
card 210's head (781eeab52), candidate this branch. Every run held `/tmp/tensorfold-model.lock` inside the Desk's Mac
GPU window, one server at a time. Other agents' builds kept the load average between 14 and 40, which moves the
controls by up to 30%; the candidate ran between them.

## Step-major holds

`bench.sh` (`tools/bench_concurrent.py --levels 1,2,4,16 --reps 3 --tokens 256 --temperatures 0 --alone`), medians
of three reps, aggregate tok/s. Control, candidate, control:

| code / chat | control | **candidate** | control again | tokens a round (all three) |
|---|---|---|---|---|
| 1 stream | 123.2 / 67.5 | **121.7 / 90.5** | 112.2 / 82.4 | 3.37 / 2.29-2.56 |
| 2 streams | 188.9 / 121.9 | **205.6 / 170.1** | 186.4 / 139.5 | 2.81 / 2.33 |
| 4 streams | 271.2 / 136.5 | **310.3 / 256.5** | 239.6 / 187.8 | 2.81 / 2.33 |
| 16 streams | 262.2 / 130.5 | **344.1 / 234.6** | 243.9 / 222.0 | 2.81 / 2.33 |

- Against the better control: +8.8% / +21.9% at two streams, +14.4% / +36.6% at four, +31.2% / +5.7% at sixteen. One
  stream keeps the solo chain (−1.2% on code, inside the controls' spread).
- Acceptance: at every depth the drafts equal the control's, so tokens a round agree cell for cell from two streams on
  (one stream's chunking varies with timing). Accepted over drafted across each session: 21,844 / 42,709 (51.1%),
  candidate 21,833 / 42,386 (51.5%), control 21,870 / 43,100 (50.7%).
- 0 failed requests; every concurrent reply equal to its run alone (3, 6, 12 and 48 of each in every session).

## Against the fastest other engine here

`engines.sh` (`--levels 1,4,64 --reps 3 --tokens 256 --temperatures 0`), end-to-end tok/s (a reply that arrives as
one text piece has no decode time, so the aggregate column reads 0 for it), medians of three:

| code / chat | TensorFold, DSpark | oMLX 0.8.2, MTP | ratio |
|---|---|---|---|
| 1 stream | **112.3 / 90.7** | 45.1 / 8.3 | 2.5x / 10.9x |
| 4 streams | **292.0 / 256.4** | 107.0 / 79.1 | 2.7x / 3.2x |
| 64 streams | **341.8 / 232.8** | 209.3 / 227.6 | 1.6x / 1.02x |

- oMLX serves its own 4-bit build (`Inkling-2-Mini-oQ4e-mtp`, about 17 GB) with MTP drafting on; TensorFold serves the
  official NVFP4/MXFP8 checkpoint. oMLX's chat reps vary 4.5x at one stream (6.9-37.7); its best rep is still 2.4x
  under ours. At 64 chat streams the two tie within the reps' spread (ours 210.9-277.3, oMLX 207.7-282.8).
- Not runnable here: Strata (no binary or weights; v0.2.19 publishes 108.9-114.5 tok/s at one stream on M5 Max),
  llama.cpp (no `llama-server`, no Inkling GGUF), mlx-lm `mlx_lm.server` (installed with Inkling support, but oMLX
  wraps the same kernels with batching, so it is the stronger of the two).

## Exactness

`TF_INKLING2_METAL_DIR` and `TF_INKLING2_DSPARK_DIR` set to the checkpoint, the `test-inkling2-metal-host` test
binary runs on the device:

- `Mini Metal DSpark sixteen-stream chain preserves solo holds at unequal depths`: depths 1-7, four streams then all
  sixteen in one hold, each equal to its own solo hold;
- `Mini Metal DSpark four prompts equal solos through a leave and join`: four greedy replies equal four solo replies
  token for token while a stream leaves and another joins;
- `Mini Metal opt-in DSpark drafting preserves target tokens and rolls back exactly`.

## Files

`cells.jsonl`: each cell of the five sessions; `bench.sh` and `engines.sh` reproduce them (`MODEL` names the
checkpoint, `OUT` the scratch folder, `OMLX` the oMLX binary).
