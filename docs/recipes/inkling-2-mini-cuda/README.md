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

Job `top-a96a7d63bedf` at `22fb3250e`, `--context 8192`, greedy, 256 tokens, 3 reps, medians; the build ran first, then
every check and both levels inside `timeout 600`. This is one run with these settings, not a benchmark.

| Streams | First token (code / chat) | Prompt (code) | Decode a stream (code / chat) | Total decode (code / chat) | Failed |
|---|---|---|---|---|---|
| 1 | 0.076 s / 0.097 s | 197.4 tok/s | 32.9 / 33.1 tok/s | 32.9 / 33.1 tok/s | 0 |
| 4 | 0.187 s / 0.207 s | 83.4 tok/s | 29.9 / 30.3 tok/s | 117.7 / 121.1 tok/s | 0 |

The 4-stream gate takes each level's better prompt: 121.1 over 33.1 tok/s, 3.66x. Every stream sends the same prompt,
so the four streams route to the same experts; mixed traffic may give a lower 4-stream rate.

Checks in the same job: the CUDA engine matched the host reference token for token, fresh and resumed prompts gave
identical logits and tokens, a prompt resumed after another prompt resumed its prefix equalled the fresh one, and two
lanes in shared rounds kept their tokens; 0 of 32 greedy tokens differed from the float64 reference (smallest reference
margin 0.11), and its planted control (no messages) differed at 6. The kernel contract passed gates 1-4 on all 5 loop
kernels with both planted controls red.

Memory, sampled after each step (not process peaks): host used (`/proc/meminfo`) was 12.3 GiB before the server,
33.5 GiB with Mini loaded for 4 streams and 33.4 GiB after both levels; the server held 20.0 GiB of GPU memory
(`nvidia-smi`).

Bandwidth limit: a decoded token reads about 5.55 GB of weights (core layers 3 passes, 8 of 128 experts a visit, the
whole unembedding), so 273 GB/s caps one stream near 49 tok/s; 32.9 tok/s is 67% of it. #4387's own engine, removed
here, gave 5.7 tok/s at one stream and 9.4 at four on the same GB10 (job `top-0759755a0515`).

## Against llama.cpp: Unsloth's UD-Q4_K_M GGUF on the same GB10

`compare.sh` builds llama.cpp b12611's CUDA `llama-server` from the Desk's staged sources, checks the GGUF's sha256
(`78fcedcf…34ae`), the gcc archive and the b12611 source tree, and serves it at the receipt's settings (4 slots of
8,192 positions, all layers on the GPU, greedy, 256 tokens, 3 reps, medians). Job `top-b77911768550` at `b65744605`;
the uncached one-stream prompt and source pin were checked by job `top-c0075bd36227` at `941dbada`. TensorFold's column
is the receipt above.

| | TensorFold (bf16 safetensors) | llama.cpp (UD-Q4_K_M) |
|---|---|---|
| Weights on disk | 20,161,820,584 bytes (18.78 GiB, with the 635,801,416-byte draft head `mtp.safetensors`) | 16,610,544,608 bytes (15.47 GiB) |
| Decode, 1 stream (code / chat) | 32.9 / 33.1 tok/s | 38.8 / 39.1 tok/s |
| Decode total, 4 streams (code / chat) | 117.7 / 121.1 tok/s | 85.9 / 87.6 tok/s |
| 4 streams over 1 | 3.66x | 2.24x |
| First token, 1 stream (code / chat) | 0.076 s / 0.097 s | 0.067 s (slot cache hit) / 0.130 s |
| Prompt, 1 stream (code, uncached) | 197.4 tok/s | 106.4 tok/s (15 tokens / one-token response time; median of 3) |
| Prompt, 4 streams (code) | 83.4 tok/s | 43.0 tok/s (cold requests only) |
| Memory, server loaded | 20.0 GiB GPU (`nvidia-smi`) | 18.1 GiB GPU (`nvidia-smi`); peak 20.57 GiB: 18.18 GPU + 2.38 resident |
| Continuation text / teacher-forced tokens vs float64 reference | 32 of 32 token picks | Same free-running text when re-tokenized / 31 of 32 token picks (differs at 5) |

`bench_concurrent.py` warms up with the same code prompt before measuring: llama.cpp's slot cache answers every
one-stream request, so its code first token is a cache hit and `prompt_ttft_tps` is null (only cold requests
qualify). The separate uncached `/completion` measurement above supplies the missing one-stream comparison; it times
the full one-token response (0.14 s) rather than first streamed token, so its client latency includes that token's
transfer. At 4 streams the prompt figure is likewise the median of cold requests only; later rounds likely reuse the
slots the first filled, so it is about one round's requests rather than a median of 3. llama.cpp decodes one stream
18% faster from 18% fewer weight bytes on disk (15% fewer without the draft head, which `serve` does not load); at 4
streams TensorFold's total is 37% higher. TensorFold's memory is a sample after loading, not
`peak-memory.py`'s process peak. Quality: llama.cpp's free-running text matches the 32-token reference continuation
when re-tokenized, but its teacher-forced token picks agree at 31 of 32 positions (different at position 5);
TensorFold's token picks agree at all 32 positions.

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
