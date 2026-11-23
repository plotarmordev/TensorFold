# Inkling 2 Mini: what makes Strata fast, and what TensorFold lacks

Card 209. A read of [Strata](https://github.com/Niko1221/Strata) (MIT, © Niko1221) at
[v0.2.20](https://github.com/Niko1221/Strata/releases/tag/v0.2.20) (`4920a3a4be`, engine `strata-inkling` 0.1.66)
and [v0.2.19](https://github.com/Niko1221/Strata/releases/tag/v0.2.19) (`8fffdbf4a3`, 0.1.65, the release that
published the numbers), set against TensorFold's Inkling 2 code on `main` and on card 188's branch
(`feat/188-inkling-2-mini-runs-whole-on-our-spark-g`, the only TensorFold path that serves Mini). No Strata code is
taken; if a follow-up card ports any of it, that card credits Strata in `THIRD_PARTY_NOTICES.md` by name and link.

## The numbers are two packs, not one configuration

From the speed table of Strata's pack card,
[strata-engine/Inkling-2-Mini-Strata](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata): thinking off,
one-stream rows with DSpark drafting on; the card states neither the engine version nor the context.

| Machine | Pack | Solo decode | c=64 total |
|---|---|---|---|
| RTX 5090 | SQ-10 (10.47 GB) | **250.3** tok/s | 3,010.4 |
| RTX 5090 | SQ-20 (19.35 GB) | 198.9 | **4,089.7** |
| DGX Spark GB10 | SQ-10 | **74.3** | 805.8 |
| DGX Spark GB10 | SQ-20 | 46.1 | **1,145.1** |

The headline pairs (250 / 4,090, 74.3 / 1,145) each mix two packs. The shipped `strata-inkling --bench`
(`engine/src/inkling2/ink_main.cpp`, lines 125–229) gives every slot a distinct synthetic 64-token prompt
(`100 + ((slot * 31 + position) % 1000)`) and its own first token from prefill logits. It is still synthetic,
not representative of any particular client traffic.

TensorFold serves Mini only on card 188's branch (bf16 trunk, the checkpoint's NVFP4 experts, `--context 8192`, greedy,
256 tokens). The 13× in the card's title is 74.3 / 5.7, against 188's first receipt (job `top-0759755a0515` at
`2301ab379`, #4387's ungraphed loop engine: 5.7 tok/s at 1 stream, 9.4 total at 4). On jschmied's #4441 engine
(graphs, fused input projections, one batched forward a round), job `top-29fb623d4dee` at `54bf2c7f0`
([receipt](https://github.com/plotarmordev/TensorFold/blob/54bf2c7f0cf3f72e5ab9ef9d4449c81a260fc6e0/docs/recipes/inkling-2-mini-cuda/README.md))
measures **32.5 tok/s at 1 stream and 117.8 total at 4**, with 0 of 32 greedy tokens off the float64 reference.
Against Strata's best GB10 solo row that is 2.3×, not 13×.

## Ceiling math on GB10 (273 GB/s)

Mini: 24 physical layers, the middle 12 looped three times (48 visits a token), 128 experts top-8, hidden 2,304,
vocab 200,058.

| Weights a token reads | Bytes / token | Plain decode ceiling |
|---|---|---|
| TensorFold, checkpoint as stored (card 188's count) | 5.55 GB | 49 tok/s |
| Strata SQ-20 (bf16 attention, SF4/MX8 experts) | ≈5.42 GB | ≈50 tok/s |
| Strata SQ-10 (Q8_ROW attention, 2–3-bit experts) | ≈3.29 GB | ≈83 tok/s |

SQ-20's 46.1 tok/s is 92% of its plain ceiling and SQ-10's 74.3 is 90% of its plain ceiling, with drafting on;
those percentages are not isolated bandwidth utilisation because drafting changes bytes per emitted token. Our no-draft
32.5 tok/s is 66% of our 49 tok/s plain ceiling: 30.7 ms a token against a ≈20 ms weight-read floor. The 5.7 → 32.5
step was graphs and batching (lever 5). The effects of utilisation and drafting on the remaining solo gap are not
separated by these receipts.

## The levers

States: **in TensorFold** (serving Mini on 188's branch), **partial**, **missing**, **n/a**. Gains are ceiling
estimates, not measurements.

### 1. W=4 on-GPU beam-tree DSpark — missing on CUDA

Strata: the drafter is embedded in the pack (3 layers, Block-8, taps 5/23/41/48 mixed by `softmax(fuse_weight)`,
a rank-256 Markov RNN head) and runs entirely on GPU. Base candidates come from a 4-bit screening plane and a radix
top-K, and the RNN head scores only those rows (`engine/src/inkling2/cuda/ink_dspark.cuh`, header lines 1–11).
`emit_draft_beam_tree` captures all 7 beam steps in one CUDA graph (W ≤ `kInkMaxBeam` = 4). `draft_beam_tree` picks
3–16 nodes by maximising expected emits over an expert-union cost `1 + 4.2·(1 − (15/16)^M)`. One 48-visit
`forward_tree_verify` pass checks the tree; device argmax IDs are copied back for the host tree walk, then
`k_ink_commit_kv_all48` / `k_ink_commit_sconv_all48` commit only the accepted path
(`ink_cuda_backend.cu::dspark_spec_step_greedy`, lines 758–830). This fast path is one stream, greedy, no penalties
(`ink_serve.h`, lines 762–810). Sampled streams use exact rejection sampling on the host.

TensorFold: the config parses Mini's MTP head and `names.zig` maps every tensor (`zig/src/families/inkling2/config.zig`
`parseMtp`, `names.zig`), but no CUDA drafter exists. 188's branch has a Metal DSpark
(`zig/src/families/inkling2/metal/dspark*.zig`, `zig/kernels/metal/inkling2/dspark.metal`); its CUDA engine
(`inkling2_loop`) has no draft path. Elsewhere TensorFold drafts on GPU with chains by default and opt-in trees
(`zig/src/families/qwen3_5/cuda_mtp_tree.zig`, `zig/src/families/deepseek_v41/draft/tree.zig`).

Would touch: a new `zig/src/families/inkling2/dspark.zig` behind `zig/src/core/lanes/drafter.zig`, and a new
`zig/kernels/cuda/inkling2/dspark.cu`, ported from the Metal kernels.
Gain: unmeasured separately. The 1.36× residual solo gap after the SQ-10 byte ratio includes both drafting and
roof utilisation (Strata's 74.4 tok/s with drafting versus TensorFold's 32.5 without); neither component is
identified without a Strata `--draft off` control at the same pack and prompt. No multi-stream drafting gain:
Strata's tree runs for one greedy stream only.

### 2. Zero-logits device-to-host — in TensorFold

Strata: greedy verify runs `head_argmax_exact` (`ink_runner.cuh`, line ~3502) through an L2-norm screen, 4,096 bf16
candidates and an exact rescore, then copies only M int32 ids back (comment: 0.14 ms on a 5090, bit-exact for M ≤ 32).
The batched and sampled path still copies full rows into pinned memory (`ink_cuda_backend.cu::emit_selected_logits`,
B × 200,058 × 4 bytes, 51 MB a step at c=64).

TensorFold: `tf_ink2_top1` (greedy) and `tf_ink2_race1` (keyed Gumbel race) already pick on device
(`zig/kernels/cuda/inkling2/ops.cu`, `zig/src/families/inkling2/ops.zig`). On 188's branch Mini draws every row on
device and downloads only int32 token ids from its ring (`mini_backend.zig`, `Mini.drawRows` and the `tok.download`
calls). The rest of the engine reads back row-best or top-k candidates only
(`zig/src/families/flashnext/cuda_draw_batch.zig`).

Would touch: nothing for Mini; it already keeps the readout to ids.
Gain: none left to take. Strata's own batched copy (51 MB a step at c=64) costs about 1 ms against TensorFold's
measured ≈108 ms c=64 step (64 streams / 590 tok/s), under 1%, so it is not a source of either engine's rate.

### 3. fp4 tensor-core MoE — partial

Strata: `ink_moe_fp4_sm120.cu` is compiled for `sm_120a`/`sm_121a`. SF4 experts (e2m1 with an e4m3 scale per 16
weights, layers 0–19, 44 of 48 visits) run on `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`, and MX8
(layers 20–23) on `kind::mxf8f6f4`. Pack bytes are already in MMA operand order ("zero bit-reordering"). `k_ink_moe_bin`
builds the grouped expert tiles on device, so it can be captured in a graph. These kernels run at M ≥ 24 rows (c=32,
c=64, prefill: `ink_moe_cuda.cuh`, lines 3251–3253). At M ≤ 16 (solo decode and every verify tree), 4-lane kernels read
each weight byte once across all rows. SQ-10/SQ-15 use INT8 tensor cores (M ≥ 16) and `dp4a` below that.

TensorFold: the native block-scaled FP4 mma exists (`zig/kernels/cuda/mma4.cuh`, the same instruction) but only
Nemotron uses it (`zig/kernels/cuda/ck_experts.cu`). 188's Inkling experts take exl3-arithmetic grouped tiles
(`zig/kernels/cuda/inkling2/experts.cu` on 188's branch). The NVFP4 expert kernels used by other families decode fp4
into bf16 mma (`zig/kernels/cuda/nvfp4_experts.cu`).

Would touch: `zig/kernels/cuda/inkling2/experts.cu` and `zig/src/families/inkling2/experts.zig` (188's branch), on
`mma4.cuh`.
Gain: none solo (Strata itself uses byte-once kernels there). At c=64: unmeasured. No receipt isolates tensor-core
from CUDA-core expert math; Strata's c=64 aggregate over its solo rate also folds in batching and loses drafting.

### 4. SQ packs — missing

Strata: `.strata` v1.1 (`engine/include/strata/artifact/strata_reader.hpp`) has a zstd JSON manifest, a trunk section
with the drafter, an expert directory with per-expert codecs (`WFmt`: S1_25…S8_K, SF4, MX8, Q8_ROW) and 4 KiB-aligned
expert bodies in kernel byte order. It is read with `fread` into 64 MiB pinned chunks (no mmap), loop-weighted experts
first. SQ-N is the pack's size in GB for Inkling: SQ-20 is lossless SF4/MX8, SQ-15 has 96.68% top-1 and SQ-10 has
92.35% (`terminal/strata_control/inkling_fit.py`).

TensorFold: broad formats (EXL3, MLX affine 2–8 bit, pre-swizzled NVFP4 in `zig/src/cuda/nvfp4.zig`), but Mini loads
its checkpoint as stored, with a bf16 trunk.

Would touch: `zig/src/families/inkling2/mini_model.zig` (188's branch), reusing `zig/src/cuda/nvfp4.zig` /
`modelopt.zig` packing for a quantised trunk.
Gain: lowers the roof. SQ-10-class bytes lift the ceiling from 49 to ≈83 tok/s (1.7×), at a measured quality cost
that a TensorFold pack has to gate on its own.

### 5. CUDA graphs — in TensorFold

Strata: `GraphKey {M, mode, taps, kv_dtype}` (`ink_runner.cuh`, lines 1757–1761). It captures the whole 48-visit body
(`emit_48_visits`) per decode shape, per verify-tree size and per batch B = 1…64, so c=64 is one `forward_batched(64)`
replay a step.

TensorFold: `zig/src/cuda/graph_cache.zig` is general. 188's head serves Mini on jschmied's #4441 engine with graphs,
fused input projections and one batched forward for every stream's rows. That move took Mini from 5.7 to 32.5 tok/s
solo and from 9.4 to 117.8 total at 4 streams (jobs `top-0759755a0515` → `top-29fb623d4dee`).

Would touch: `zig/src/families/inkling2/mini_model.zig` (188's branch, `Model.capture`), to key graphs by tree size once
lever 1 lands.
Gain: taken (5.7×). The last third of the roof (30.7 ms → ≈20 ms a token, up to 1.5×) is profiling, not graphs.

### 6. 64-slot batching — partial

Strata: 64 slots, 64-token KV pages, FP8 KV from 16 slots, and one decode token from every slot each round
(`engine/src/inkling2/ink_batch.h`, `ink_serve.h`). Only 8 of 48 visits keep global KV, so 64 resident slots are cheap.

TensorFold: lanes and paged KV are general (`zig/src/core/lane_host.zig`, `zig/src/core/kv_pool.zig`). Mini on 188
already serves 64 batched streams (job `top-975292aaf4c0`), but holds bf16 KV per slot.

Would touch: `zig/src/families/inkling2/mini_model.zig` for FP8 KV at higher residency and context.
Gain: rides on levers 3 and 5. On TOP, Strata's SQ-10 does 382.5 tok/s at 64 streams, 5.1× its solo rate (below).

## Strata's engine on TOP, next to ours: measured

Strata's own linux-aarch64 build ran as shipped, with no host changes, through `swarm gpu-run` (job
`top-edfe387b00e5`, this branch at `f88916b9e`).

- The Desk staged the engine tarball (sha256 `a01f0e1f…`, from the release's `SHA256SUMS.txt`) and the SQ-10 pack
  (sha256 `ac4d452a…`, pinned to Hugging Face revision `04996a1e2c45…`) in TOP's pilot folder
  `strata-inkling-pack/`, as [`recipes/strata-inkling/strata-stage.sh`](recipes/strata-inkling/strata-stage.sh) does.
- [`recipes/strata-inkling/strata-job.sh`](recipes/strata-inkling/strata-job.sh) runs the engine's `--bench` at 1, 4
  and 64 streams (64-token prompt, 256 generated tokens) and `--selfcheck` (PASS, `sm_121`, `sm120_tc=1`). The binary
  needs glibc 2.38, newer than the job image's, so it runs under the loader and libraries of the Arm GNU 14.3
  toolchain already staged for `cuda-build.sh`, unpacked in `/work`.

| GB10 decode, tok/s | 1 stream | 4 streams total | 64 streams total | Source |
|---|---|---|---|---|
| Strata 0.1.66, SQ-10 pack, DSpark on | **74.4** | **97.1** (24.3 a stream) | 382.5 | job `top-edfe387b00e5` |
| TensorFold, checkpoint as stored, no drafting | **32.5** | **117.8** (29.6 a stream) | 589.8–593.0 (different client protocol) | jobs `top-29fb623d4dee` at `54bf2c7f0`; `top-975292aaf4c0` at `505ffbc48` |
| TensorFold, #4387's loop engine | 5.7 | 9.4 | not run | job `top-0759755a0515`, card 188 at `2301ab379` |

These benches are not matched: Strata's built-in `--bench` uses distinct synthetic 64-token prompts and 256 generated
tokens per slot. TensorFold's `tools/bench_concurrent.py` uses code and chat prompts over HTTP; its 64-stream receipt
(card 211's
[`inkling-2-mini-cuda/README.md`](https://github.com/plotarmordev/TensorFold/blob/502666b8836abaa2d6cad0bc478c20572a8e46ae/docs/recipes/inkling-2-mini-cuda/README.md#hill-climb-and-limits),
job `top-975292aaf4c0` at `505ffbc48`) uses `--parallel 64 --context 4096`, 128 generated tokens, and medians of
three repeats. Its control arms were 589.8, 593.0 and 592.1 tok/s; the table gives their range. Prompt routing,
context and run length prevent attributing the 64-stream difference to an engine lever alone. The gap, lever by
lever:

- **Solo, Strata leads 2.29×.** Its 74.4 reproduces the published 74.3. SQ-10 reads ≈3.29 GB a token against our
  5.55 GB, so the smaller pack (lever 4) raises the plain ceiling by 1.69×. The residual 1.36× is **not decomposed**:
  Strata's draft-on 74.4 sits at 90% of its ≈83 tok/s plain ceiling while our no-draft 32.5 sits at 66% of ours;
  these ratios alone cannot distinguish improved weight-read utilisation from drafting. Run Strata's supported
  `--draft off` control on the same SQ-10 pack and bench before assigning a gain to lever 1.
- **Four streams, TensorFold leads 1.21×.** Strata's DSpark fast path is one greedy stream only; at 4 it falls back
  to plain batched decode and loses 3.06× per stream relative to solo, where 188's graphed batch loses 9%.
  Graphs and batching (levers 5 and 6) are where TensorFold is already ahead on these runs.
- **64 streams:** Strata's SQ-10 reaches 382.5, not the published 1,145, which is the SQ-20 pack's row. TensorFold's
  589.8–593.0 control is 1.54–1.55× higher on its different HTTP workload; this is not a matched comparison.

## Windows on ARM: nobody ships Inkling 2 Mini fast there

- Strata ships no Windows ARM64 build. v0.2.19 and v0.2.20 have `windows-x86_64` engines only. `install.ps1` refuses
  any non-AMD64 Windows ("a Windows on ARM PC … has no engine for ARM processors"), and
  [issue #4579, "ARM64 Windows (RTX Spark)"](https://github.com/Niko1221/Strata/issues/4579), is open.
- The only Strata build on the N1X is a community fork,
  [strata-n1x-Zackyzq](https://github.com/zackyangzeqi-spec/strata-n1x-Zackyzq), for DeepSeek-V4.1-Flash, not Inkling.
- CUDA itself runs there. CUDA 13.4 ships Windows ARM64 nvcc, cudart and cuBLAS, and NVIDIA's
  [RTX Spark guide](https://build.nvidia.com/rtx-spark/running-llamacpp) downloads a prebuilt Windows ARM64
  llama.cpp. Ollama and LM Studio ship WoA CUDA builds too. None carries Strata's levers, and whether any runs
  Inkling 2 is unverified here.
- What the laptop still needs: Mini's CUDA engine built `aarch64-windows` with MSVC ARM64 as nvcc's host compiler
  (only gcc 14 has built these Inkling kernels), then the GB10 exactness and WDDM allocation-delta checks on an N1X.
  TensorFold **already implements** WDDM-pool admission via `cuMemGetInfo_v2` (`docs/WINDOWS.md`, lines 167–173);
  this needs a Mini-specific device receipt, not a new admission path. The kernels target the same `sm_121`. No N1X
  run of Mini exists from anyone. Card 188's recipe README has the build and test details.

## Follow-up cards, in order

For 207 (GB10):

1. **A quantised trunk (a TensorFold pack).** Pack the bf16 attention to 8-bit rows and the experts below 4 bits,
   gated on top-1/KL against the checkpoint. Touches `mini_model.zig` and `zig/src/cuda/nvfp4.zig`. Gain: a roof from
   49 to ≈83 tok/s (the bytes ratio, 1.69× of the measured 2.29× solo gap), at every stream count.
2. **CUDA DSpark beam tree for Mini.** Port 188's Metal DSpark to CUDA with W ≤ 4, adaptive tree nodes, one verify
   sweep and on-device commit, under the conformance ledger's tree = chain = plain rule. Touches new
   `zig/src/families/inkling2/dspark.zig` and `zig/kernels/cuda/inkling2/dspark.cu`. Gain: unmeasured; the
   residual 1.36× solo ratio after the pack-byte ceiling is shared with weight-read utilisation (30.7 ms versus a
   ≈20 ms floor), not a measured or hard upper bound on drafting. Strata `--draft off` at the same pack and bench
   is required to isolate it; no drafting gain at 4 streams.
3. **Native fp4 grouped experts at M ≥ 24, and FP8 KV for 64 resident streams.** Put Mini's NVFP4 experts on
   `mma4.cuh`'s `mxf4nvf4` with device-built expert tiles; 64 lanes already run (job `top-975292aaf4c0`), so what
   remains is Strata's FP8 KV from 16 slots in place of Mini's bf16 per-slot KV, to hold 64 streams at longer
   context. Touches `zig/kernels/cuda/inkling2/experts.cu`, `mini_model.zig`. Gain: unmeasured on matched workloads;
   TensorFold's 589.8–593.0 at c=64 already exceeds Strata SQ-10's 382.5 in the unmatched receipt, so those rates
   cannot bound the fp4 gain.

For 202 (RTX Spark laptop, Windows-native, N1X-tuned):

1. **Build and validate Mini's CUDA engine `aarch64-windows`.** MSVC ARM64 as nvcc's host, CUDA 13.4
   `windows-arm64` redistributables; run GB10 exactness and WDDM allocation-delta checks on the N1X. Pool admission
   is already implemented. Gain: enables Mini, but tok/s is unmeasured on N1X.
2. **Bring 207's quantised trunk and CUDA DSpark tree to the N1X.** Measure its own bandwidth roof: GB10-class
   silicon does not guarantee 273 GB/s sustained in a laptop. Gain: the trunk's byte ratio raises a bandwidth roof
   up to 1.69× (before quality gating); drafting is unmeasured and depends on N1X acceptance and utilisation.
3. **Bring 207's grouped-fp4 experts and 64-lane path to N1X, then tune.** Measure the M ≥ 24 tensor-core crossover,
   the WDDM pool's resident stream count, and tree size by acceptance and bytes per emitted token. Gain: unmeasured
   at laptop batch sizes; the GB10 64-stream receipts use different workloads and provide no transferable multiplier.
   This would be the first Windows ARM64 Inkling engine anyone ships.
