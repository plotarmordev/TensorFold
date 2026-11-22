# Inkling 2 Mini on TOP's GB10

`run-script.sh` runs thinkingmachines/Inkling-2-Mini (revision `ed40e7da62`) whole on TOP's Spark through the Desk's
GPU door. The job has no network, so the checkpoint must already sit in the pilot folder as
`models/Inkling-2-Mini`; the script refuses and lists what `/assets` holds when it does not.

The server is #4387's engine (`inkling2.native`, model type `inkling2_loop`) with jschmied's three load and
attention fixes from his #4387 comments; #4441's engine (`inkling2.mini_native`) rides along for its float64
reference. The script fails on any of: `tf-cuda-test inkling2-serving` (#4387's checks), `tf-cuda-test
inkling2-mini-ref` (32 greedy tokens equal to the float64 host reference), that reference's planted control staying
green, or a failed stream in `tools/bench_concurrent.py` at 1 and 4 streams.

## RTX Spark (N1X): what differs and what must change

I see the same chip and memory model as the GB10: sm_121 and one unified pool, so the `-Dsm=121` kernel images are
the same. What differs is the host. RTX Spark runs Windows on Arm, with WSL2 as the
Linux route. From #1010 (`5b1f10812`) and #2107:

- Native Windows builds as `aarch64-windows` with VS 2022 Build Tools' ARM64 C++ tools (`vcvarsall arm64`) as nvcc's
  host compiler and the CUDA 13.4 `windows-arm64` redistributables (winget has no arm64 installer). Admission reads the
  WDDM pool through `cuMemGetInfo_v2`, and `nvidia-smi`'s NPU row is filtered out.
- Under WSL2 `/proc/meminfo` is WSL's RAM (about 30 GiB of 128 GB), not the GPU's 96 GB WDDM pool; with
  `WSL_DISTRO_NAME` set, admission reads the CUDA pool instead (`TENSORFOLD_UNIFIED_HOST_COUNTS` overrides it).
- For Mini, I need three things before it builds and serves there: the Inkling CUDA kernels compiled by MSVC's
  front end (only nvcc with gcc 14 has built them so far), one load of the 18.9 GiB single-shard checkpoint
  through Windows' threaded reads, and the GB10's exactness checks repeated on the N1X. No N1X run of Mini exists yet.
