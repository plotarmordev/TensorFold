# Inkling 2 Mini: what makes Strata fast, and what TensorFold lacks

Card 209. A read of [Strata](https://github.com/Niko1221/Strata) (MIT, © Niko1221) at
[v0.2.20](https://github.com/Niko1221/Strata/releases/tag/v0.2.20) (commit `a34979fc94`, engine `strata-inkling` 0.1.66)
and [v0.2.19](https://github.com/Niko1221/Strata/releases/tag/v0.2.19) (commit `c6f6427d9d`, 0.1.65, the release that
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
(graphs, fused input projections, one batched forward a round), job `top-16bffb6ca094` at `9f542e6fa`
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

States: **in TensorFold** (serving Mini on 188's branch), **missing**, **n/a**. Gains are ceiling estimates, not
measurements.

### 1. W=4 on-GPU beam-tree DSpark — missing on CUDA

Strata: the drafter is embedded in the pack (3 layers, Block-8, taps 5/23/41/48 mixed by `softmax(fuse_weight)`,
a rank-256 Markov RNN head) and runs entirely on GPU. Base candidates come from an 8-bit screening plane of the
unembedding (`unembed_screen.q8`, scored by `k_dspark_screen_dp4a7`; the header comment's "4-bit" is stale) and a
radix top-K, and the RNN head scores only those rows (`engine/src/inkling2/cuda/ink_dspark.cuh`, lines 9–10 and
1543–1550).
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

Would touch: a new `zig/src/families/inkling2/dspark.zig` behind `zig/src/core/lanes/drafter.zig` and a new
`zig/kernels/cuda/inkling2/dspark.cu`. Only the proposal block can follow the Metal kernels: 188's Metal drafter is
chain-only (`metal/dspark.zig`: `absorbFn` returns `TreesNotBuilt` for tree rows, its `factsFn` leaves `trees` and
`tree_nodes` at 0, and its vtable sets no `tree` callback), and Mini's CUDA backend takes chains only
(`mini_backend.zig`: `verify` refuses held drafts, parents and masks; `keep` refuses a shorter path with
`Inkling2MiniNoRollback` and a non-identity path with `ChainsOnly`). The beam tree is new work, specified under
follow-up 2.
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

### 3. fp4 tensor-core MoE — missing

Strata: `ink_moe_fp4_sm120.cu` is compiled for `sm_120a`/`sm_121a`. In SQ-20, SF4 experts (e2m1 with an e4m3 scale per
16 weights, layers 0–19, 44 of 48 visits) run on `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`, and
MX8 (layers 20–23) on `kind::mxf8f6f4`. Pack bytes are already in MMA operand order ("zero bit-reordering").
`k_ink_moe_bin` builds the grouped expert tiles on device, so it can be captured in a graph. SQ-10 and SQ-15 keep SF4
experts too, mixed with S_K codecs (SQ-10: S2/S3_K + SF4, SQ-15: SF4 + S3/4/5_K, per the
[runner header](https://github.com/Niko1221/Strata/blob/v0.2.20/engine/src/inkling2/metal/ink_metal_runner.h),
line 4). `launch_ink_moe_dispatch_only` in
[`ink_moe_cuda.cuh`](https://github.com/Niko1221/Strata/blob/v0.2.20/engine/src/inkling2/cuda/ink_moe_cuda.cuh)
dispatches on each layer's codec flags, not on the pack: at M ≥ 24 rows (c=32, c=64, prefill; lines 3252–3253) a
layer's SF4/MX8 experts take these fp4/fp8 tensor-core kernels (line 3292) and its S_K experts take the INT8
tensor-core `k_ink_moe_sk_tc` (lines 3323–3328), in the same call. So at c=64 SQ-10's SF4 experts run on this lever's
fp4 path; only its S_K experts use INT8. Below M = 24 (solo decode and every verify tree of up to 16 rows) SF4/MX8 run
on CUDA cores: `k_ink_moe_gu_sf4_mx8<1>` reads each weight once for one row, and `<4>` rereads it once per four-row
group; the S_K and GGUF codecs use `dp4a` the same way.

TensorFold: the native block-scaled FP4 mma exists (`zig/kernels/cuda/mma4.cuh`, the same instruction) but only
Nemotron uses it (`zig/kernels/cuda/ck_experts.cu`). On 188's branch Mini's experts go through the shared grouped
path: `mini_visit.zig`'s `moe` calls `Grouped.decode` / `Grouped.prompt` (`zig/src/cuda/experts.zig`), which send
layers 0–19 to `Nvfp4Experts.run` / `runPrompt` (`zig/kernels/cuda/nvfp4_experts.cu`, with the soft-capped SwiGLU
gate-up `tf_nvfp4_softcap` from `experts_mx.zig`'s `Softcap`) and layers 20–23 to `Mxfp8Experts.run`
(`zig/kernels/cuda/fp8_experts.cu`). Both kernels widen the stored codes to bf16 pairs (`fp4pair`, `e4m3x2` in
`common/fp_pairs.cuh`) and sum on bf16 mma, at every row count. `zig/kernels/cuda/inkling2/experts.cu` is the full
model's EXL3 path; Mini does not run it.

Would touch: `zig/kernels/cuda/nvfp4_experts.cu` (an `mxf4nvf4` entry on `mma4.cuh`, keeping the soft caps) behind
`Nvfp4Experts.run` / `runPrompt` in `zig/src/cuda/experts.zig`, the same for `fp8_experts.cu` / `Mxfp8Experts.run` on
`kind::mxf8f6f4`, and `mini_visit.zig`'s `moe` to pick it when a call has M ≥ 24 rows (c=32, c=64 decode, prompts).
Gain: none solo. At M = 1 every expert weight is read once whatever the math unit, and Strata itself keeps CUDA cores
below M = 24. At c=64: unmeasured. No receipt isolates tensor-core from CUDA-core expert math; Strata's c=64
aggregate over its solo rate also folds in batching and loses drafting.

### 4. SQ packs — missing

Strata: `.strata` v1.1 (`engine/include/strata/artifact/strata_reader.hpp`) has a zstd JSON manifest, a trunk section
with the drafter, an expert directory with per-expert codecs (`WFmt`: S1_25…S8_K, SF4, MX8, Q8_ROW) and 4 KiB-aligned
expert bodies in kernel byte order. It is read with `fread` into 64 MiB pinned chunks (no mmap), loop-weighted experts
first. SQ-N is the pack's size in GB for Inkling: SQ-20 is lossless SF4/MX8, SQ-15 has 96.68% top-1 and SQ-10 has
92.35% on test prompts ("How close is it to the original?" on the pack card,
[strata-engine/Inkling-2-Mini-Strata](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata)).

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
solo and from 9.4 to 117.8 total at 4 streams (jobs `top-0759755a0515` → `top-16bffb6ca094`).

Would touch: `zig/src/families/inkling2/mini_model.zig` (188's branch, `Model.capture`), to key graphs by tree size once
lever 1 lands.
Gain: taken (5.7×). The last third of the roof (30.7 ms → ≈20 ms a token, up to 1.5×) is profiling, not graphs.

### 6. 64-slot batching — in TensorFold

Strata: 64 slots, 64-token KV pages, and one decode token from every slot each round
(`engine/src/inkling2/ink_batch.h`, `ink_serve.h`). Only 8 of 48 visits keep global KV, so 64 resident slots are cheap.

TensorFold: lanes and paged KV are general (`zig/src/core/lane_host.zig`, `zig/src/core/kv_pool.zig`). Mini on 188
already serves 64 batched streams in one forward a round (job `top-975292aaf4c0`).

Would touch: nothing.
Gain: taken. On TOP, TensorFold's c=64 control (589.8–593.0 tok/s) is 18× its solo rate; Strata's SQ-10 does 382.5,
5.1× its own (below).

### 7. FP8 KV — missing

Strata: paged KV in FP16 or FP8 E4M3 (`engine/src/inkling2/ink_batch.h`, `KvDtype`: 32 KiB a token of global KV in
FP16, 16 KiB in FP8, 160 / 80 MiB for the full sliding windows), chosen per server by `kv_dtype`; INT8/4-bit K8V8 and
K8V4 go further.

TensorFold: Mini holds bf16 KV per slot (`mini_model.zig`: 40 sliding rings, 8 whole caches). A sliding ring covers
the 1,024-token window plus a 512-row prompt chunk, 1,536 rows after 64-row alignment (`kernels.zig`, `ringFor`). With
8 KV heads × 128 (Mini's `config.json`), each visit stores 4 KiB of K and V a token in bf16.

Would touch: `zig/src/families/inkling2/mini_model.zig` (cache allocation) and the attention reads in
`zig/src/families/inkling2/mini_visit.zig`.
Gain: cache bytes halve. A stream at context L allocates 4 KiB × (40·1,536 + 8·L): 496 MiB at L = 8,192 in bf16
(31 GiB for 64 streams; 243's laptop serve reports 0.49 GiB a stream), 248 MiB in FP8, so twice the streams or
context fit the same memory. Decode reads only the live rows, 4 KiB × (40·min(L, 1024) + 8·L) a stream: at the end
of Strata's bench (L = 320) that is 60 MiB a stream, 3.75 GiB for 64 streams a step, of which FP8 saves 1.9 GiB,
more at longer context.

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
| TensorFold, checkpoint as stored, no drafting | **32.5** | **117.8** (29.6 a stream) | 589.8–593.0 (different client protocol) | jobs `top-16bffb6ca094` at `9f542e6fa`; `top-975292aaf4c0` at `505ffbc48` |
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
- **64 streams:** Strata's SQ-10 reaches 382.5, not the published 1,145, which is the SQ-20 pack's row; it is also
  only 47% of Strata's published same-pack SQ-10 row (805.8), unexplained here. Card 242 owns the SQ-20 and SQ-10
  remeasurement. TensorFold's 589.8–593.0 control is 1.54–1.55× higher on its different HTTP workload; this is not
  a matched comparison.

## Card 251: Unsloth's llama.cpp, Mia's Derisked server, and the lever list

### What Unsloth's build is

Unsloth's "1.9× faster, 125 tok/s on an RTX PRO 6000" (2026-11-23) is not a branch. `unslothai/llama.cpp` `master`
(`a5aa5bacbc2b`) holds only a pin list, `scripts/unsloth/pr-set.json`: 43 PR commits that CI merges onto a ggml-org
tag. The tree it ships is the source asset `llama.cpp-source-b12651-mix-c7059d6.tar.gz` (sha256 `dedb1c00…7bb0`) of
release [`b12651-mix-c7059d6`](https://github.com/unslothai/llama.cpp/releases/tag/b12651-mix-c7059d6): mainline
[b12651](https://github.com/ggml-org/llama.cpp/releases/tag/b12651) plus the pins. Unsloth's docs table
(`tg1024`, UD-Q4_K_XL, F16 KV, DSpark **off**) gives **125.24 vs mainline b12649's 81.35** on the RTX PRO 6000 (1.54×)
and 110.77 vs 68.00 on a B200; the 1.9× headline is a Ryzen AI Max row (25.65 vs 13.74). Diffing the shipped tree
against b12651's source adds seven CUDA files, all graph-pattern fusions matched in `ggml-cuda.cu`'s
`ggml_cuda_try_fuse`, each with a `GGML_CUDA_DISABLE_*_FUSION` switch and a `test-backend-ops` case.

| Unsloth pin (unslothai/llama.cpp) | What it does | Unsloth's stated gain | In TensorFold's Mini engine (188) |
|---|---|---|---|
| [#386](https://github.com/unslothai/llama.cpp/pull/386) `210fc2653` | banded flash attention for the relative bias; one block stack per prefix, no repeat at T = 1 | 7,328 → 6,653 kernels a token, tg128 +8–10% | yes: `tf_ink2_rel_table` once, attention reads it (`attention.cu`) |
| [#401](https://github.com/unslothai/llama.cpp/pull/401) `6c0e84ddf` | q/k/r L2 clip (sqr, sum_rows, sqrt, div, clamp, mul) as one kernel (`clip-rows.cu`) | → 5,723 kernels | yes: the clip is inside `tf_ink2_prep` |
| [#402](https://github.com/unslothai/llama.cpp/pull/402) `0bfd204bd` | SiTU gate, 9 nodes → 1 (`situ.cu`) | → 5,350; tg128 +27% with #400/#401 | yes: soft caps fused into the expert gate-up (`tf_nvfp4_softcap`, `tf_ink2_gateup`) |
| [#411](https://github.com/unslothai/llama.cpp/pull/411) `7e633d60b` | short conv: concat, state store, SSM_CONV, dynamic taps, residual add, scale, ≈16 kernels → 1 (`conv-chain.cu`) | 950 → 192 kernels; −11.5% tg128 on UD-Q4_K_XL without it | yes: `tf_ink2_conv` (state, dynamic taps, residual modes in one kernel) |
| [#412](https://github.com/unslothai/llama.cpp/pull/412) `874c88cad` | tanh·mul±add, router norm, add + rms_norm (`elem-fuse.cu`) | −606 kernels, tg128 +5–7% | partly: the router's top-k and its weight norm are one kernel (`tf_ink2_topk`); the loop controller's gate is three (`tf_ink2_loop_gate_down`, `_lin`, `_mix` in `loop.zig`) |
| [#413](https://github.com/unslothai/llama.cpp/pull/413) `260b1ffa4` | relative bias read inside the vector flash attention (`fattn-rel.cu`) | −240 kernels, tg128 +2–6% | yes (as #386) |
| [#414](https://github.com/unslothai/llama.cpp/pull/414) `f766457c8` | depth mix: rms_norm, concat, q·stack, softmax, mix → 1 (`depth-mix.cu`) | −300 kernels, tg128 +2–3% | yes: `tf_ink2_mix` / `tf_ink2_mix_rows` |
| [#400](https://github.com/unslothai/llama.cpp/pull/400) `a5667a9c0` | `mul_mat_vec_f` up to 8 rows a block | pp512 +36%, tg flat | n/a: TensorFold's mix is one kernel |
| [#415](https://github.com/unslothai/llama.cpp/pull/415) `a7c4e14de`, [#416](https://github.com/unslothai/llama.cpp/pull/416), [#418](https://github.com/unslothai/llama.cpp/pull/418) | matvec instead of mma at 2–8 columns, one q8_1 copy shared by same-input matmuls, flat concat | 4 streams 1.21–1.35× | n/a as named (TensorFold has no same-input matmul pairs or concat nodes to collapse); no pin-off run here, so their share of Unsloth's 4-stream code total (121.9–139.8 vs our 118.7) is unmeasured → card 258 |
| [#417](https://github.com/unslothai/llama.cpp/pull/417), [#421](https://github.com/unslothai/llama.cpp/pull/421) | MoE matvec 4 rows (cc 12.x keeps 2); one cache entry per looped expert | host-expert runs only | n/a on GB10 (unified memory, every expert resident) |
| [#405](https://github.com/unslothai/llama.cpp/pull/405) `b5ed9f284` | loads Unsloth's padded-vocab DSpark drafter | 126.6 vs 102.1 greedy, B200 | no: lever 1 (cards 236, 246) |

Unsloth's gain is kernel count: about 7,300 kernels a token down to about 3,400 on 1.8 TB/s cards, where Mini's
5 GB of weights take ≈3 ms and a few µs a kernel dominate. TensorFold's Mini engine was written fused from the start;
its one-row decode launches about 1,200 kernels a token ([INFERENCE]: a count of 236's source, about 20 a plain visit
and 29 a looped one), already below Unsloth's. On GB10 the same kernel time sits under a 21–23 ms weight-read floor
(5.55 GB at the 242 GB/s a copy measures, against the 273 GB/s spec), so Unsloth's levers are worth less there than
on its cards, and TensorFold has most of them.

### GB10, the same GGUF

[`recipes/inkling-2-mini-llamacpp/job.sh`](recipes/inkling-2-mini-llamacpp/job.sh) builds mainline b12651 and
Unsloth's tree from source with card 232's flags plus `-fa on`, and serves 232's UD-Q4_K_M GGUF (sha256
`78fcedcf…34ae`, no second download) at 232's settings (4 slots, all layers on the GPU, greedy, 256 tokens, 3 reps,
`tools/bench_concurrent.py` code and chat prompts). The Desk stages both source tarballs with
[`stage.sh`](recipes/inkling-2-mini-llamacpp/stage.sh).

| Decode tok/s, code / chat prompts | 1 stream | 4 streams total | Source |
|---|---|---|---|
| TensorFold (bf16 checkpoint, 188's engine) | 32.9 / 32.8 | 118.7 / 120.1 | job `top-b77911768550` (232) |
| llama.cpp b12611, UD-Q4_K_M | 38.8 / 39.1 | 85.9 / 87.6 | job `top-b77911768550` (232) |
| llama.cpp b12651, UD-Q4_K_M | 38.4 / 39.2 | 85.5 / 88.8 | job `top-d561fdb4ceb9`, this branch at `a0efea99d` (waits: jobs `top-28403507400a`, `top-87b11998bb03`, `top-68cf8da0fc84`, `top-2faa590d03a5` ran before staging) |
| Unsloth b12651-mix-c7059d6, UD-Q4_K_M | **44.7 / 45.3** | **139.8 / 117.1**; code 121.9 on the review's same-head rerun | job `top-d561fdb4ceb9`; review rerun |
| *RTX PRO 6000, Unsloth's docs (UD-Q4_K_XL, tg1024)* | *125.24 (Unsloth) / 81.35 (mainline b12649)* | — | Unsloth |

Measured on GB10: Unsloth over mainline is **+16% solo** (45.3 / 39.2 chat; expected above: the removed kernel time
is a smaller share under a ≈21 ms weight-read floor than under ≈3 ms on the 1.8 TB/s cards, so 1.54× shrank to
1.15–1.17×) and **1.43–1.63× / 1.32× at 4 streams** (code, across the two runs / chat). The 4-stream code total
moved from 139.8 to 121.9 between two runs of the same build, so one run does not fix it; neither run switched pins
off, so which pins carry the 4-stream gain is not measured here. Against TensorFold's rows, Unsloth's build leads
solo (45.3 vs 32.9 chat, on ≈3.3 GB of weights a token against our 5.55 GB) and on 4-stream code (121.9–139.8 vs
118.7); TensorFold still leads 4-stream chat (120.1 vs 117.1). Solo, llama.cpp reads fewer weight bytes a token than
our bf16 trunk does; this card's MXFP8 trunk (below) takes TensorFold to 41.7 solo and 144.8–149.7 at 4 streams.

### Mia's 45.8 tok/s: Derisked is not a faster model

[Blackfrost-AI/Inkling-2-Mini-Derisked](https://huggingface.co/Blackfrost-AI/Inkling-2-Mini-Derisked) (revision
`31ec2edd`) is a refusal edit: per-block directions on layers 9–18 applied to the weights, "native vendor quantization,
unmodified", the MTP module byte-identical to upstream, served on vLLM with DSpark (7 draft tokens). Its one 19.5 GB
shard has the checkpoint's layout and bytes a token, so it cannot read fewer bytes than ours. Mia's
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) (`86f972e2`, `server/collectors/DecodeBench.js`) streams
`/v1/chat/completions` at temperature 0.7, top_k 20, thinking on; a 1.39× lead at one stream on the same bytes points
to drafting in her TensorFold build ([INFERENCE]: her server's flags are not published). For scale: Unsloth's plain
llama.cpp build, no drafting, decodes 45.3 tok/s solo on the same GB10 — but on UD-Q4_K_M weights (≈3.3 GB read a
token from a 16.6 GB file), so byte-identical comparisons still need drafting. Matching it on our GB10 needs our
server with DSpark (card 236's branch) and sparkDash's Node collectors staged for the job; that is follow-up 4 below,
with sparkDash's prefill sweep (4k–128k), which TensorFold has no GB10 receipt for.

### Inco's Splash: a 4-bit GGUF and a block-diffusion drafter

[Splash 1.7.2](https://github.com/incoai/splash/blob/main/docs/release-notes/1.7.2.md) (Apache-2.0, a Metal engine,
2026-11-24) is Inco's "200+ tok/s on an M5 Max" (its release lists 213 on a 32-core M5 Max, pre-release build; 113 and
149 on 16- and 20-core M5 Pros). Its two levers:

- **Weights:** the default is Unsloth's `UD-Q4_K_M` (16.61 GB, card 232's GGUF), not the release's bf16 trunk; the
  release's BF16 attention and head load as Q8_0, which its notes measure within a mean KL of 0.0024 and the same top
  token at 98.7% of 1,933 teacher-forced positions. This is card 252's lever, and that number is its first quality
  evidence.
- **Drafter:** [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), a 0.49B
  block-diffusion draft (5 layers, an 8-row block, top-16 candidates a row and a low-rank pairwise selector that traces
  one path) reading the target's rows at applications 5, 23 and 41. Its notes give 6.30 accepted tokens a cycle
  (greedy, 512-token answers, UD-Q4_K_M) and its card 4.8–5.0 for code and 3.9 for chat when sampled on an H200, where
  MTP/DSpark gives 2.95–3.66.

Its decode work matches TensorFold's and Unsloth's fused glue (one call for a layer's three mixer gates, a block mix
per query, one relative-attention pass a layer, the FFN convolution in the MoE combine). No Mac GPU window was open
for this card, so Splash is unmeasured on our M5 Ultra (DSpark there: 122.5 code / 91.9 chat, plain ≈62); card 255
measures it and DFlash2's acceptance against DSpark's.

### Levers, marked

| Lever | Source | In TensorFold (Mini, 188) | Expected GB10 gain, 1 stream | Card |
|---|---|---|---|---|
| Fused glue: conv chain, clip, SiTU, depth mix, rel bias in attention | Unsloth #386, #401, #402, #411, #413, #414 | in | taken | — |
| Controller gate fusion and dependent launches (PDL off for Mini: `qmmf.zig` `pdl` false) | Unsloth #412 | missing | 2–3 ms of 30.4: +7–11% ([INFERENCE], no per-kernel GB10 profile) | 253 |
| Small-batch kernel choice, q8_1 reuse, flat concat | Unsloth #415, #416, #418 | n/a as named (no same-input matmul pairs or concat nodes to collapse) | none at 1 stream; at 4 streams Unsloth's code total (121.9–139.8 over two runs) beats our 118.7, with no pin-off run to tie it to these pins | 258 |
| MoE cache dedupe, 4-row MoE matvec | Unsloth #417, #421 | n/a on GB10 | — | — |
| W=4 beam-tree DSpark | Strata 1; Unsloth #405 | missing on CUDA (chain drafter on 236's branch) | unmeasured; Mia's 45.8 at the same bytes suggests ≈1.4× | 236, then 209's follow-up 2 |
| Block-diffusion drafter (DFlash2) | Splash | missing | unmeasured on our hardware; sampled on an H200, 4.8–5.0 (code) and 3.9 (chat) tokens a cycle against MTP/DSpark's 2.95–3.66 ([INFERENCE]: a larger gain than DSpark's) | 255 |
| Zero-logits device-to-host | Strata 2 | in | taken | — |
| fp4 tensor-core MoE at M ≥ 24 | Strata 3 | missing | none at 1 stream; c=64 unmeasured | 209's follow-up 3 |
| Smaller trunk bytes (SQ packs) | Strata 4 | **built here (card 251), opt-in:** `TF_INKLING2_TRUNK=mxfp8`; bf16 stays the default | measured: 32.9 → **41.7** tok/s (+27%) at 1 stream, 117.2–121.0 → **144.8–149.7** (+24%) at 4; NVFP4 ≈52 (+58%) unbuilt | 252 (NVFP4 and the quality gate) |
| CUDA graphs | Strata 5 | in | taken | — |
| 64-slot batching | Strata 6 | in | taken | — |
| FP8 KV | Strata 7 | missing | none at short context; twice the resident streams | 209's follow-up 3 |

**The biggest missing lever, trunk bytes, is built in this card.** Card 188's Mini engine is merged into this branch
(`4051c7372`); `TF_INKLING2_TRUNK=mxfp8` rounds the bf16 attention projections (the fused `qx`/`all` rows, `wq`…`wo`)
and the unembedding to MXFP8 at load in `mini_weights.zig` (e4m3 codes, a UE8M0 scale per 32 inputs chosen so no code
saturates) and runs them on `qmmf.zig`'s existing `mxfp8` lane kernels. One job, one build, both arms
([`recipes/inkling-2-mini-cuda/trunk-ab.sh`](recipes/inkling-2-mini-cuda/trunk-ab.sh), job `top-751b5c791bc7` at
`77d6f94bd`, 188's settings: 4 slots of 8,192, greedy, 256 tokens, 3 reps, medians):

| Decode tok/s, code / chat | 1 stream | 4 streams total |
|---|---|---|
| bf16 trunk (default) | 32.9 / 32.7 | 117.2 / 121.0 |
| MXFP8 trunk | **41.7 / 41.7** | **144.8 / 149.7** |

That is +27% solo, short of the +34% estimated from bytes (24.0 ms a token against 30.4), and +24% at 4 streams.
It passes Unsloth's build at 4 streams (139.8 and 121.9 code, 117.1 chat) but not solo (45.3). Quality: with MXFP8,
1 of 32 greedy tokens differs from the float64 bf16 reference (position 24, where the reference's margin is 0.11);
bf16 gives 0 of 32, and llama.cpp's UD-Q4_K_M agrees at 31 of 32 teacher-forced picks (232). One prompt, so this
does not show the error is harmless; the switch stays opt-in until card 252's KL gate passes.
A second job on the same code (`top-5e3d39db7aed` at `376d52d0f`, same settings) repeats it: bf16 32.9 / 33.0 solo and
117.2 / 121.0 at 4 streams, MXFP8 41.7 / 41.2 and 145.4 / 149.7, and again 1 of 32 tokens differs.

### Follow-up cards from 251 (252–254 and 258 start after 188 merges; 255 needs a Mac GPU window)

1. **Card 252: NVFP4 trunk and head for Mini on CUDA, and the quality gate for both.** MXFP8 is built here (41.7 tok/s,
   1 of 32 tokens differ); add `qmmf`'s `fp4` mode for the same rows, then a top-1/KL gate on the float64 reference over
   many prompts decides whether either becomes the default. Gain: ≈+58% for NVFP4 over bf16 (estimate), +27% measured for MXFP8.
2. **Lossless alternative to 252, if its quality gate fails:** a 12-bit coded bf16 GEMV and a certified greedy head screen
   ([dsgb10](https://github.com/blaesus/dsgb10), MIT). Gain: ≈+15–20%, exact.
3. **Card 253: controller-gate fusion and PDL for Mini.** Merge the loop gate's three kernels and the router-plan launches,
   set `pdl` on Mini's lanes; a GB10 per-kernel profile first. Gain: +7–11% ([INFERENCE]).
4. **Card 254: sparkDash on our GB10, decode and prefill.** Stage sparkDash `86f972e2`'s collectors and Node, run Net's
   settings (prose, 256 tokens, temperature 0.7, top_k 20, thinking on, 1/2/4 streams; prefill 4k–128k) against our
   server with and without DSpark, beside Mia's 45.8 / 69.9 / 102.9 and 2,877…1,919.
5. **Card 255: Splash and DFlash2 on our M5 Ultra.** Serve Splash 1.7.2 with `UD-Q4_K_M` and its DFlash2 draft in a
   Mac GPU window (after the owner agrees to the install), beside DSpark's 122.5 / 91.9, and compare accepted tokens a
   cycle. Gain: unmeasured; DFlash2's published acceptance is 1.3–1.6× DSpark's.
6. **Card 258: close Unsloth's 4-stream code-prompt lead.** Its build's code total at 4 streams (139.8 in job
   `top-d561fdb4ceb9`, 121.9 on the review's same-head rerun) beats our 118.7 while our chat total still leads (120.1
   vs 117.1). First repeat both arms enough times to fix the spread, then run Unsloth's build with and without pins
   #415, #416 and #418 to measure their share; profile TensorFold's 4-stream code step, land what transfers, and
   re-measure beside these rows.

## Windows on ARM: Inkling builds exist; none is Strata's or tuned for the N1X

- Strata ships no Windows ARM64 build. v0.2.19 and v0.2.20 have `windows-x86_64` engines only. `install.ps1` refuses
  any non-AMD64 Windows ("a Windows on ARM PC … has no engine for ARM processors"), and
  [issue #4579, "ARM64 Windows (RTX Spark)"](https://github.com/Niko1221/Strata/issues/4579), is open.
- The only Strata build on the N1X is a community fork,
  [strata-n1x-Zackyzq](https://github.com/zackyangzeqi-spec/strata-n1x-Zackyzq), for DeepSeek-V4.1-Flash, not Inkling.
- Inkling 2 Mini runs on llama.cpp's shipped `win-cuda-13.5-arm64` pre-releases.
  [b12600](https://github.com/ggml-org/llama.cpp/releases/tag/b12600) adds Mini (`model : Inkling 2 Mini`, #32694;
  [b12406](https://github.com/ggml-org/llama.cpp/releases/tag/b12406) was the full Inkling 2 only);
  [b12611](https://github.com/ggml-org/llama.cpp/releases/tag/b12611)
  ([#32704](https://github.com/ggml-org/llama.cpp/pull/32704)) fixes Mini's looped NVFP4 expert scales; and
  [b12641](https://github.com/ggml-org/llama.cpp/releases/tag/b12641)
  ([#32799](https://github.com/ggml-org/llama.cpp/pull/32799), `ignore_merges` for inkling2) brings tokenizer
  parity, the first fully correct Mini build with experts on the GPU.
  [b12642](https://github.com/ggml-org/llama.cpp/releases/tag/b12642)
  ([#32807](https://github.com/ggml-org/llama.cpp/pull/32807)) fixes stale expert copies in Mini's shared looped
  layers when experts stay on the host (`--n-cpu-moe`), so it is the first fully correct one there. llama.cpp has
  its own parallel batching; this study did not audit which of its kernels Mini takes on Blackwell, and no N1X tok/s
  for it is known here.
- TensorFold's own Mini engine builds, passes its checks and serves natively on the N1X, unmerged (card 243's
  [`docs/recipes/inkling-2-mini-cuda/README.md`](https://github.com/plotarmordev/TensorFold/blob/df6af4786/docs/recipes/inkling-2-mini-cuda/README.md)
  at `df6af4786`). Run 1 at `b2ddc3de` gives build exit 0, `inkling2-serving` 3/3 and `inkling2-mini-ref` 0 of 32
  tokens differing; run 2 at `2a37918af` decodes **32.2** tok/s at one stream and **117.0** total at four (server
  `done` lines, 256 tokens a stream), against GB10's 32.5 / 117.8, both without drafting.
- What the laptop still needs: its own measured bandwidth roof, a merge of 243, and the GB10 levers below brought
  over and tuned. TensorFold **already implements** WDDM-pool admission via `cuMemGetInfo_v2` (`docs/WINDOWS.md`,
  lines 167–173). The kernels target the same `sm_121`.

## Follow-up cards, in order

For 207 (GB10):

1. **A quantised trunk (a TensorFold pack).** Pack the bf16 attention to 8-bit rows and the experts below 4 bits,
   gated on top-1/KL against the checkpoint. Touches `mini_model.zig` and `zig/src/cuda/nvfp4.zig`. Gain: a roof from
   49 to ≈83 tok/s (the bytes ratio, 1.69× of the measured 2.29× solo gap), at every stream count.
2. **CUDA DSpark beam tree for Mini.** No tree exists to port: 188's Metal DSpark and Mini's CUDA backend are both
   chain-only. Build (a) the CUDA proposal block from the Metal kernels as a chain drafter first, then (b) a W ≤ 4 beam
   tree on device: per-step top-W expansion over the block's logits, node selection by expected emits against the
   expert-union cost (Strata's `draft_beam_tree`). Contracts to change:
   - Drafter (`drafter.zig`): the CUDA drafter reports `Facts.trees` / `tree_nodes`, accepts `Absorb.rows`, and
     implements the `tree` callback returning drafts and parents; `metal/dspark.zig` does the same and drops
     `TreesNotBuilt` once Metal follows.
   - Verify (`mini_backend.zig` `verify`): accept held drafts, `parents` and masks instead of refusing them;
     `mini_model.zig` `Chain` / `forward` and `mini_visit.zig` `Rows.stage` take per-row parents and a tree
     attention mask (positions by depth).
   - Accepted path (`mini_backend.zig` `keep`): accept a path shorter than the window and non-identity row indices;
     commit only those rows to KV and short-conv state (Strata's `k_ink_commit_kv_all48` /
     `k_ink_commit_sconv_all48`), which needs the conv states to keep the pre-window rows they now discard.

   All under the conformance ledger's tree = chain = plain rule. Gain: unmeasured; the
   residual 1.36× solo ratio after the pack-byte ceiling is shared with weight-read utilisation (30.7 ms versus a
   ≈20 ms floor), not a measured or hard upper bound on drafting. Strata `--draft off` at the same pack and bench
   is required to isolate it; no drafting gain at 4 streams.
3. **Native fp4 grouped experts at M ≥ 24, and FP8 KV.** Give `Nvfp4Experts.run` / `runPrompt`
   (`zig/kernels/cuda/nvfp4_experts.cu`, `zig/src/cuda/experts.zig`) an `mxf4nvf4` entry on `mma4.cuh` with the soft
   caps, and `Mxfp8Experts.run` (`fp8_experts.cu`) a `mxf8f6f4` one, picked by `mini_visit.zig`'s `moe` at M ≥ 24;
   then FP8 KV in `mini_model.zig` (lever 7). Gain: the fp4 math is unmeasured on matched workloads (TensorFold's
   589.8–593.0 at c=64 already exceeds Strata SQ-10's 382.5 in the unmatched receipt, so those rates cannot bound
   it); FP8 KV halves cache bytes, ≈1.9 GiB of a c=64 step's reads at L ≈ 320 and twice the resident streams or
   context.

For 202 (RTX Spark laptop, Windows-native, N1X-tuned):

1. **Merge Mini's CUDA engine `aarch64-windows`.** 243 already builds, passes the exactness checks and decodes 32.2 /
   117.0 tok/s at 1 / 4 streams on the N1X (run 2); what remains is the WDDM allocation-delta check and a merge.
   Gain: the measured N1X baseline lands on main, at GB10's no-draft rate.
2. **Bring 207's quantised trunk and CUDA DSpark tree to the N1X.** Measure its own bandwidth roof: GB10-class
   silicon does not guarantee 273 GB/s sustained in a laptop. Gain: the trunk's byte ratio raises a bandwidth roof
   up to 1.69× (before quality gating); drafting is unmeasured and depends on N1X acceptance and utilisation.
3. **Bring 207's grouped-fp4 experts and 64-lane path to N1X, then tune.** Measure the M ≥ 24 tensor-core crossover,
   the WDDM pool's resident stream count, and tree size by acceptance and bytes per emitted token. Gain: unmeasured
   at laptop batch sizes; the GB10 64-stream receipts use different workloads and provide no transferable multiplier.
   No shipped Windows ARM64 Inkling build cited here (llama.cpp's, 243's unmerged one) publishes N1X-tuned numbers.
