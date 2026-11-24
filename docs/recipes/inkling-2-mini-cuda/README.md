# Inkling 2 Mini on TOP's GB10

`run-script.sh` runs thinkingmachines/Inkling-2-Mini (revision `ed40e7da62`) whole on TOP's Spark through the Desk's
GPU door. The job has no network, so the checkpoint must already sit in the pilot folder as `Inkling-2-Mini` (or
`models/Inkling-2-Mini`); the script refuses and lists the Inkling configs under `/assets` when it does not.

The server is jschmied's #4441 engine (`inkling2.native`, model type `inkling2_loop`, on his #4364 and #4365): graphs,
fused input projections and one batched forward for every stream's rows in a round. #4387's checks run against it.
The script fails on any of: `tf-cuda-test kernel-contract "inkling2 loop"` (the served message and gate kernels
against FP64 references, with their planted controls), `tf-cuda-test inkling2-serving` (#4387's checks), `tf-cuda-test inkling2-mini-ref` (32
greedy tokens equal to the float64 host reference), that reference's planted control staying green, a failed stream
in `tools/bench_concurrent.py` at 1 and 4 streams, or 4 streams giving under 1.5 times one stream's total tok/s. The
receipt also sets one stream's decode against GB10's bandwidth limit: the weight bytes a token reads (from the
checkpoint's headers) over 273 GB/s.

## Receipt: TOP's GB10, one capped run

Job `top-16bffb6ca094` at `9f542e6fa`, `--context 8192`, greedy, 256 tokens, 3 reps, medians. This is one run with
these settings, not a benchmark.

| Streams | First token (code / chat) | Prompt (code) | Decode a stream | Total decode | Failed |
|---|---|---|---|---|---|
| 1 | 0.077 s / 0.038 s | 195.9 tok/s | 32.5 tok/s | 32.5 tok/s | 0 |
| 4 | 0.189 s / 0.092 s | 82.6 tok/s | 29.6 tok/s | 117.8 tok/s (3.62x) | 0 |

Memory, sampled after each step (not process peaks): host used (`/proc/meminfo`) was 10.9 GiB before the server,
31.6 GiB with Mini loaded for 4 streams, and the same after both levels; the server held 20.0 GiB of GPU memory
(`nvidia-smi`).

Bandwidth limit: a decoded token reads about 5.55 GB of weights (core layers 3 passes, 8 of 128 experts a visit, the
whole unembedding), so 273 GB/s caps one stream near 49 tok/s; 32.5 tok/s is 66% of it. #4387's own engine, removed
here, gave 5.7 tok/s at one stream and 9.4 at four on the same GB10 (job `top-0759755a0515`).

## RTX Spark (N1X): what differs and what must change

I see the same chip and memory model as the GB10: sm_121 and one unified pool, so the `-Dsm=121` kernel images are
the same. What differs is the host. RTX Spark runs Windows on Arm, with WSL2 as the
Linux route. From #1010 (`5b1f10812`) and #2107:

- Native Windows builds as `aarch64-windows` with VS 2022 Build Tools' ARM64 C++ tools (`vcvarsall arm64`) as nvcc's
  host compiler and the CUDA 13.4 `windows-arm64` redistributables (winget has no arm64 installer). Admission reads the
  WDDM pool through `cuMemGetInfo_v2`, and `nvidia-smi`'s NPU row is filtered out.
- Under WSL2 `/proc/meminfo` is WSL's RAM (about 30 GiB of 128 GB), not the GPU's 96 GB WDDM pool; with
  `WSL_DISTRO_NAME` set, admission reads the CUDA pool instead (`TENSORFOLD_UNIFIED_HOST_COUNTS` overrides it).
- For Mini on native Windows, I need the Inkling CUDA kernels compiled with MSVC as nvcc's host front end (only gcc 14
  has built them so far) and one load of the 18.9 GiB single-shard checkpoint through Windows' threaded reads.
- For Mini under WSL2, I need neither: it is the Linux build with gcc 14. Only the pool reading above differs.
- Either way, the GB10's exactness checks must be repeated on the N1X. No N1X run of Mini exists yet.

## Hill-climb and limits

`hillclimb.sh` runs `setup.sh`'s build, checkpoint digest check and exactness gates (logged as `build: N s`), then
runs each measurement as its own job under `timeout 580` (logged as `measure NAME: N s`): the no-draft limits from
`ceiling.py`, then fresh servers in the order control, `TF_NVFP4_DECODE_WARPS=1`, control, `TF_QMMF_HALF_BELOW=0`,
control. An arm cut off at 580 s lands in the ledger and ends the run with its exit status. It measures one same-prompt
stream and 64 same-prompt and mixed-prompt streams; each cell requires a valid timing from every request. The 64
different prompts in [`prompts-64.json`](prompts-64.json) have sha256
`027e6b85a51f007f83c7b2002067aa650f5a5c65c83bb95c180038de5dd930f6`; the ledger records the path and hash.

Job `top-975292aaf4c0` at `505ffbc48` (#4441's engine, `--parallel 64 --context 4096`, greedy, 128 tokens; medians
of 6 reps at one stream and 3 at 64). Gates: the serving checks passed, 0 of 32 greedy tokens off the float64
reference (the planted control caught 6 of 32), and no request failed.

| Arm (checkpoint as stored, no draft; same-prompt only) | 1 stream, tok/s | 64 streams total, tok/s |
|---|---:|---:|
| control | 32.6 | 589.8 |
| `TF_NVFP4_DECODE_WARPS=1` | 32.9 | 596.9 |
| control | 32.9 | 593.0 |
| `TF_QMMF_HALF_BELOW=0` | 32.8 | 593.3 |
| control | 32.8 | 592.1 |

Neither lever clears its controls by more than the 3% noise (+0.7% and +0.1% at 64 streams): both rejected; the
defaults stay. **No mixed-prompt result has been measured** for this receipt. The no-draft limit of 49.2 tok/s at one
stream applies only to this 5.55 GB/token stored checkpoint at the GB10's 273 GB/s specification (the job cannot
open the DRM card for the bandwidth probe); it does not bound Strata's smaller packs. The 64-stream estimate under
uniform routing (534.8 tok/s) is not a bound: actual routing shares experts across rows.

Strata v0.2.19's published GB10 rows are for **different packs**: SQ-10 gives 74.3 tok/s at one stream and 805.8
total at 64; SQ-20 gives 46.1 at one and 1,145.1 total at 64. Card 209's TOP run of Strata 0.1.66 on the staged
SQ-10 pack gave 74.4 at one and 382.5 at 64 with its built-in `--bench` (job `top-edfe387b00e5`), against our
32.6–32.9 and 589.8–593.0 on the stored checkpoint; prompt lengths (64 vs code/chat), generated lengths (256 vs
128), client protocols and pack sizes differ, so these are **not matched engine comparisons**. Strata SQ-20 has
no same-session TOP measurement in this receipt. Card 247's first HTTP run of Strata on the SQ-10 pack (at
`1e17bb6c`: 53.4, 79.3 and 269.3 tok/s at 1, 4 and 64 streams) returned mostly empty replies, so it is not a fair
bar yet, and the published rows have not been reproduced on this GB10. Neither published pack nor the one-stream bar
is beaten by this engine; Mini CUDA has no draft path (`takes_drafter = false`). Card 209 proposes a CUDA DSpark tree
on #4128's kernels, then fp4 tensor-core experts at 64 streams and a smaller pack. Cards 215 and 219 own that work,
but their future results are not evidence for this card's bar.
