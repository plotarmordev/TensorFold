# Inkling 2 Mini on TOP's GB10

`run-script.sh` runs thinkingmachines/Inkling-2-Mini (revision `ed40e7da62`) whole on TOP's Spark through the Desk's
GPU door. The job has no network, so the checkpoint must already sit in the pilot folder as `Inkling-2-Mini` (or
`models/Inkling-2-Mini`); the script refuses and lists the Inkling configs under `/assets` when it does not.

The server is jschmied's #4441 engine (`inkling2.native`, model type `inkling2_loop`, on his #4364 and #4365): graphs,
fused input projections and one batched forward for every stream's rows in a round. #4387's checks run against it.
Upstream, jschmied closed #4441 on 23 Nov as superseded: Mini on CUDA goes through #4387's engine there (#4640, then
#4697, #4718, #4728 and #4729 for speed). This branch is local; an upstream claim must move to that stack first.
The script fails on any of: `tf-cuda-test kernel-contract "inkling2 loop"` (the served message and gate kernels
against FP64 references, with their planted controls), `tf-cuda-test inkling2-serving` (#4387's checks), `tf-cuda-test inkling2-mini-ref` (32
greedy tokens equal to the float64 host reference), that reference's planted control staying green, a failed stream
in `tools/bench_concurrent.py` at 1 and 4 streams, or 4 streams giving under 1.5 times one stream's total tok/s. The
receipt also sets one stream's decode against GB10's bandwidth limit: the weight bytes a token reads (from the
checkpoint's headers) over 273 GB/s.

## Receipt: TOP's GB10, one capped run

Job `top-bde5d508e589` at `fbbd43cbe`, `--context 8192`, greedy, 256 tokens, 3 reps, medians. This is one run with
these settings, not a benchmark.

| Streams | First token (code / chat) | Prompt (code) | Decode a stream (code / chat) | Total decode (code / chat) | Failed |
|---|---|---|---|---|---|
| 1 | 0.076 s / 0.036 s | 196.8 tok/s | 32.9 / 32.8 tok/s | 32.9 / 32.8 tok/s | 0 |
| 4 | 0.189 s / 0.090 s | 82.3 tok/s | 29.9 / 30.2 tok/s | 118.7 / 120.1 tok/s | 0 |

The 4-stream gate takes each level's better prompt: 120.1 over 32.9 tok/s, 3.65x. Every stream sends the same prompt,
so the four streams route to the same experts; mixed traffic may give a lower 4-stream rate.

Checks in the same job: the CUDA engine matched the host reference token for token, fresh and resumed prompts gave
identical logits and tokens, and two lanes in shared rounds kept their tokens; 0 of 32 greedy tokens differed from
the float64 reference (smallest reference margin 0.11), and its planted control (no messages) differed at 6. The
kernel contract passed gates 1-4 on all 5 loop kernels with both planted controls red.

Memory, sampled after each step (not process peaks): host used (`/proc/meminfo`) was 10.4 GiB before the server,
30.7 GiB with Mini loaded for 4 streams, and the same after both levels; the server held 20.0 GiB of GPU memory
(`nvidia-smi`).

Bandwidth limit: a decoded token reads about 5.55 GB of weights (core layers 3 passes, 8 of 128 experts a visit, the
whole unembedding), so 273 GB/s caps one stream near 49 tok/s; 32.9 tok/s is 67% of it. #4387's own engine, removed
here, gave 5.7 tok/s at one stream and 9.4 at four on the same GB10 (job `top-0759755a0515`).

## Against llama.cpp: Unsloth's UD-Q4_K_M GGUF on the same GB10

`compare.sh` builds llama.cpp b12611's CUDA `llama-server` from the Desk's staged sources, checks the GGUF's sha256
(`78fcedcf…34ae`) and serves it at the receipt's settings (4 slots of 8,192 positions, all layers on the GPU,
greedy, 256 tokens, 3 reps, medians). Job `top-b77911768550` at `b65744605`; TensorFold's column is the receipt above.

| | TensorFold (bf16 safetensors) | llama.cpp (UD-Q4_K_M) |
|---|---|---|
| Weights on disk | 20,161,820,584 bytes (18.78 GiB) | 16,610,544,608 bytes (15.47 GiB) |
| Decode, 1 stream (code / chat) | 32.9 / 32.8 tok/s | 38.8 / 39.1 tok/s |
| Decode total, 4 streams (code / chat) | 118.7 / 120.1 tok/s | 85.9 / 87.6 tok/s |
| 4 streams over 1 | 3.65x | 2.24x |
| First token, 1 stream (code / chat) | 0.076 s / 0.036 s | 0.067 s / 0.130 s |
| Prompt, 4 streams (code) | 82.3 tok/s | 43.0 tok/s |
| Memory, server loaded | 20.0 GiB GPU (`nvidia-smi`) | 18.1 GiB GPU (`nvidia-smi`); peak 20.57 GiB: 18.18 GPU + 2.38 resident |
| Greedy vs the float64 reference, 32 tokens | 32 of 32 | 32 of 32 free-running; 31 of 32 teacher-forced (differs at 5) |

llama.cpp's one-stream prompt speed is blank: `bench_concurrent.py` recorded none for it at that level. llama.cpp
decodes one stream 18% faster from 18% fewer weight bytes; at 4 streams TensorFold's total is 37% higher.
TensorFold's memory is a sample after loading, not `peak-memory.py`'s process peak. Quality: both engines reproduce
the float64 reference's 32 greedy tokens; teacher-forced on the reference's prefix, the GGUF picks another token once
(position 5), where TensorFold picks none.

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
  has built them so far) and one load of the 18.2 GiB text shard (`model-00001-of-00001.safetensors`) through Windows'
  threaded reads.
- For Mini under WSL2, I need neither: it is the Linux build with gcc 14. Only the pool reading above differs.
- Either way, the GB10's exactness checks must be repeated on the N1X. No N1X run of Mini exists yet.
