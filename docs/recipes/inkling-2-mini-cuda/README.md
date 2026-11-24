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

`dspark.sh` checks Mini's DSpark draft head (`mtp.safetensors`) alone. It builds, then runs `tf-cuda-test inkling2-dspark`:
the head absorbs 1,100 seeded context rows, makes seven greedy drafts with each pick fed back on the device, and
checks every step's logits within 2% RMS of the float64 host reference. The planted context-blind reference must fail.
At `816d47102` (job on TOP's GB10, build 525 s, measure 113 s), the worst step was 0.41% and 0 of 7 picks differed.
The planted run reached 29.8%, with 6 of 7 picks differing. Absorb plus draft took 37 ms, including the 1,100-row
absorb; this is not a draft-step timing.

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

`hillclimb.sh` runs `setup.sh`'s build and exactness gates, then the no-draft limits from `ceiling.py`, then fresh servers in
the order control, `TF_NVFP4_DECODE_WARPS=1`, control, `TF_QMMF_HALF_BELOW=0`, control. It measures one same-prompt
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
no same-session TOP measurement in this receipt. Neither published pack nor the one-stream bar is beaten by this
engine; Mini CUDA has no draft path (`takes_drafter = false`). Card 209 proposes a CUDA DSpark tree on #4128's kernels,
then fp4 tensor-core experts at 64 streams and a smaller pack. Cards 215 and 219 own that work, but their future
results are not evidence for this card's bar.

## Mia's DGX Spark recipe next to ours

[MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold) at
`6061fa89a34d2a245bc4a230f04948f312347a5e` (24 Nov 2026; Apache-2.0 in `LICENSE`, TensorFold's MIT notice in
`NOTICE`) is a Docker recipe around TensorFold at `dc2ca8a1` (PR #4441, C4) plus seventeen patches, built in
`nvcr.io/nvidia/pytorch:26.07-py3` with Zig 0.17.0 (`zig build fatbins native -Doptimize=fast
-Dnative-fp-decode=true`, kernels embedded for sm_80/86/120/121), `libtfgrammar.so` over xgrammar v0.2.8
(`97787376`), and libjpeg-turbo/libwebp for JPEG and WebP. `-Doptimize=fast` is Zig 0.17's ReleaseFast.

**Weights.** Its default is the upstream `thinkingmachines/Inkling-2-Mini` at revision `7a67ae23`: shard
`65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`, `mtp.safetensors`
`7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`, ModelOpt NVFP4 experts in layers 0–19, MXFP8 in
20–23, bf16 dense and MTP. `RERISKED=1` instead serves Blackfrost's `Inkling-2-Mini-Derisked` from a local folder the
recipe never downloads: a behaviourally modified derivative with most refusals removed, whose deployment kit targets
vLLM only and whose MTP module alone matches upstream's. Its screenshots and numbers are that derivative served by
another engine, not this recipe on upstream weights; only the upstream shard counts here.

**Settings and harness.** `scripts/config.sh`: `CONTEXT=262144`, `PARALLEL=8`, `KV_DTYPE=fp8`, `DRAFTS=1` (DSpark
`--drafter`), `VISION=1`, `TF_INKLING2_ATTN_FP8=1` (patch 0012, lossy), `TF_INKLING2_PREFILL_TC=1` (patch 0015, lossy),
`MAX_TOKENS=32768`, temperature 1.0/top_p 1/top_k 0; its published decode table is headed `temp 0.7 · thinking on ·
top_k 20 · DSpark drafts`, so the table is sampled, not the config defaults. The harness is sparkDash
(`MiaAI-Lab/sparkDash` at `86f972e2`), Mia's dashboard: its decode-benchmark panel runs concurrency 1, 2, 3, 4, 6, 8 at
500 tokens a stream. The repository's `tools/bench_streams.py` describes the same rows but imports `tools/bench.py`,
which the repository does not ship, and sends temperature 0. The published table therefore cannot be regenerated from
the recipe's files and is treated below as Mia's deployment result, not as a reproducible baseline.

**The seventeen patches against `c99eb630d`** (`partial`: the mechanism exists without the patch's path; `M`: missing):

| Patch | What it does | Ours |
|---|---|---|
| 0001 | Drafted verify windows: exact conv rollback, DSpark taps | partial — `Model.rewind`/`keep` restore suffixes, `facts.exact_width` is 1, no taps |
| 0002 | DSpark head wired into serving (`--drafter`) | partial — `dspark.zig:Head.draft` stands alone; `takes_drafter = false`, `draft` returns `NoDraftHead` |
| 0003 | `--kv-cache fp8`: E4M3 K/V with an fp32 scale | M — `kernels.zig:Cache` is bf16 only |
| 0004 | Inkling reply parser (reasoning/content split, tool calls) | M |
| 0005 | Vectorized relative table, layer-batched loop mix, row-tiled gates, batched DSpark holds | M |
| 0006 | FP8 draft unembedding/phi, keyed sampled drafts, verify/hold costs | M |
| 0007 | Two warps an expert unit, split gate-up/down | M |
| 0008 | Certified lazy unembedding head, warp top-k, z-block cap, FP8 draft layers | M |
| 0009 | Fused router, six-projection GEMM, fused sliding prompt attention | M |
| 0010 | Numeric `reasoning_effort`, Inkling "none means off", reasoning field | M — compatibility |
| 0011 | Double-buffered bf16 decode K/V prefetch (`cp.async`) | M |
| 0012 | Opt-in FP8 attention projections | M — lossy |
| 0013 | CUDA grammar masks and Inkling structured output/tool choice | partial — `verify` returns `StructuresUnsupported` |
| 0014 | Prompt rows 512 → 4,096, 64-entry wide CUDA-graph cache | partial — one-row decode graphs exist, `prompt_rows = 512` |
| 0015 | Opt-in native block-scaled prefill tensor cores | M — lossy prefill, decode unchanged |
| 0016 | Native CUDA `inkling2_vision` | partial — `opening` refuses images |
| 0017 | Server fixes: grammar freed on a failed stream, tool-call guard, limits | M — not speed |

**Missing transferable gains.** Exact paths: 0014 (4,096-row prompt passes; at `prompt_rows = 512` every prompt chunk
reads each expert again), 0009 and 0005 (fused router and projections, tiled loop mix), 0007 (two-warp expert split),
0008 (certified lazy head, warp top-k), 0011 (`cp.async` decode prefetch), 0001–0002 (verify window and serving the head
we already have kernels for), 0003 (fp8 KV: a memory lever first, lossy). Lossy and opt-in: 0012 (98.26% teacher-forced
agreement by her README) and 0015 (96.3% agreement, 732/734 on her set). 0004, 0010, 0013, 0016 and 0017 are parser,
compatibility, structured output, vision and server work. None is a demonstrated gain on our tree; a gain the paired run
prices gets its own card, and this page does not change our measured baseline.

**The paired run.** `mia-compare.sh` builds both arms from the staged trees and serves them serially on one GB10 with the
same checkpoint; `mia-compare.py` drives each with greedy decode at 1/2/4/8 streams on the prose and code prompts (256
tokens, median of 3), prefill from ~4k to ~128k tokens with a fresh text a repetition (median by prefill seconds), and six
agreement prompts (five short, one ~4k) compared by token sha and text sha.

- Same checkpoint: the shard and MTP SHA-256 gates above; another revision stops the job.
- Matched settings: `--context 131072 --parallel 8`, bf16 KV, drafts off, greedy (temperature 0, top_p 1, seed 0),
  thinking at the server default. Her defaults cannot be matched on this head: our Mini CUDA has no fp8 KV and refuses
  `--drafter`, so the pair holds what both can serve and her levers stay quoted from her README.
- Her table's settings would compare a greedy arm against a sampled one, so they are not the pair's settings.
- Staged by the Desk (the job has no network, so no Docker image): `mia-inkling/` (her repository at `6061fa89`),
  `tf-dc2ca8a1/` (TensorFold's tree), `Inkling-2-Mini/` (revision `7a67ae23`), and the toolchain archives
  `../minimax-h3-cuda/toolchain-stage.sh` stages. The patches are applied to the staged tree and built with the
  Dockerfile's flags.
- Duration: two builds (about 9 min each, as our receipt's 525 s build was) and two measurement phases; our 128k prefill
  is unmeasured and, at the receipt's 199 tok/s prompt rate, about 11 min at 512-row prompt passes, so the prefill phase
  defaults to a 1,800 s bound (`PREFILL_S`; decode 900 s, agreement 600 s), above the one-GPU-slot 600 s rule, and
  the Desk serializes the job.

Receipt: pending the GPU window (`swarm gpu-run plotarmordev/TensorFold <head>
docs/recipes/inkling-2-mini-cuda/mia-compare.sh`).
