# Inkling 2 Mini on TOP's GB10

`run-script.sh` runs thinkingmachines/Inkling-2-Mini (revision `ed40e7da62`) whole on TOP's Spark through the Desk's
GPU door. The job has no network, so the checkpoint must already sit in the pilot folder as `Inkling-2-Mini` (or
`models/Inkling-2-Mini`); the script refuses and lists the Inkling configs under `/assets` when it does not.

The server is jschmied's #4441 engine (`inkling2.native`, model type `inkling2_loop`, on his #4364 and #4365): graphs,
fused input projections and one batched forward for every stream's rows in a round. #4387's checks run against it.
The script fails on any of: `tf-cuda-test inkling2-serving` (#4387's checks), `tf-cuda-test inkling2-mini-ref` (32
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

Memory (`/proc/meminfo`, host used): 10.9 GiB before the server, 31.6 GiB with Mini loaded for 4 streams, and the
same after both levels. GPU (`nvidia-smi`): 20.0 GiB held by the server.

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
