# MiniMax-H3-Image speed, round 1: three independent sweeps, ranked

This round looked for ways to make one MiniMax-H3-Image picture at least 1.5x faster end to end on an M5 Ultra
(256 GB) running TensorFold's native Metal port. Three sweeps ran at the same time, each on a different model family
and in its own scratch checkout of our main that held neither this report nor any branch of it:

| Sweep | Question | Model (provider) | Ran, UTC, 13 Nov |
|---|---|---|---|
| Architecture | Where one picture's time goes in the text encoder, transformer and decoder, and what the maths allows | Claude Opus 5.5 (Anthropic) | 02:48:19-02:57:53 |
| History | What our own past speed rounds (GLM, Nemotron, DeepSeek, Drama, Ornith, the H3 video model) teach an image model | GPT-6 Astra (OpenAI) | 02:48:21-03:01:36 |
| Outside | Papers and other runners, as raw material, for what nobody has combined or done on a Mac | GLM-5.3 (Z.ai) | 02:48:30-02:56:22 |

All three were started at 02:48:16 and overlapped from 02:48:30 to 02:56:22. Their proposals met only in this
synthesis. Earlier drafts of this report rested on sweeps that did not run together; they are replaced, and nothing
below comes from them. Agreement between sweeps is the confidence signal. A claim a sweep made about the code was
checked before it was used here; one that failed the check is turned down below. Nothing was run on a GPU for this
round.

## What this was made on

- **TensorFold 1.2.7**, our main at `eb7ac7a6a`, which every sweep read and which is merged into this branch.
- **The maintainer's open H3-Image pull requests**, read by their descriptions at these heads: #2766 `3b73752aa49e`,
  #2775 `ad173c6231fd`, #2778 `c555f283d852`, #2779 `28067fa16942`, #2780 `325bd04876d7`, #2781 `fa0e45c77b68`,
  #2796 `2334d60f5465`, #2808 `8e31db228219`, #2810 `32e520d08f58`.
- **Measured, inherited:** card 34's stage times, card 35's cold read-ahead runs and cards 50 and 51's resident
  server, all on this M5 Ultra and recorded in [the Metal recipe](../recipes/minimax-h3-image-metal.md);
  h3-image-mlx's 356.84 s on an M5 Pro. **Estimated:** every gain and cost in this report.

## Where a picture's time goes

Measured on this M5 Ultra, card 34's prompt, seed 42, seconds:

| Path | 2048 x 2048 | 4096 x 4096 | Source |
|---|---|---|---|
| One step, GPU | 1.83 | 9.80-9.86 | card 34 |
| Six steps | 11.1 | 59.4 | card 34 |
| One-shot tools, each mapping its weights from disk | 71.4 | 189.4, of which 36-66 s writing the PNG after the decoder's tiles | card 34 |
| Cold, every weight file evicted, read ahead | 61 (prompt 29, transformer 28 of which load 12.7, decoder 4) | 123 (13, 78 of which 16.9, 32) | card 35 |
| Warm | 12.70-12.77, one resident process (prompt 0.22, steps 11.1, decoder and PNG 1.4) | about 100, one read-ahead run, not yet split by stage | cards 51, 35 |

The architecture sweep's derivation from the port's shapes:

- **A step is compute-bound on the tensor units.** 183 / 1,018 TFLOP a step at 2048 / 4096 (linear layers 87% /
  62%, attention the rest). Fitting the two measured step times gives about 98 TFLOP/s on the linear layers and
  114 TFLOP/s in attention: attention already runs at the GEMMs' rate.
- **Memory round trips are small.** About 8.7 GB a block at 4096, mostly hidden under compute; fusing the SwiGLU
  and the heads scatter would save at most 2-3% of a step.
- **The text encoder is 0.21-0.22 s warm**, against a floor near 0.06 s. Cold, it reads at about 1.1 GB/s, a third
  of the transformer's 3.8 GB/s from the same disk.
- **The decoder's GPU work is small** (1.17 / 4.2 s of tiles). At 4096 the time after the tiles is the single-threaded
  zlib stream in `moss_image/png.zig` (Zig's level-6 deflate, filter 0), 36-66 s in card 34's one-shot runs and
  about 35 s inferred for a warm picture.

So an exact 1.5x is not available inside the steps. It is available in the 4096 PNG tail and in the cold reads.
Inside the steps, large gains are lossy and opt-in.

Facts the proposals rest on, checked in the code here:

- **Each block's MLP output is added in the next block's first norm** (`block.zig`, `run`). Leaving a block out
  needs an explicit residual boundary, not a skipped call.
- **The production GEMM already reads its operands straight from device memory** (`prefill/gemm_nax.metal`:
  `const device T*` operands, the only barrier `mem_none`), one 64 x 128 x 256 geometry for every projection.
- **The step heads pin the schedule at six steps**; each step's trunk output feeds its own head.
- **Attention is dense both ways.** Text rows take each step's timestep line, so no K/V is constant across steps.

## Guardrails every proposal meets

- **Not already someone's.** Card 40's design (commit `bcc2f15c`: the resolution cascade, routed attention in the
  last steps, the decoder in 512-pixel tensor-unit tiles), [main's ideas page](../recipes/minimax-h3-image-ideas.md)
  (the cascade, pooled far-field keys `far=pool`, the GPU and SME2 row split), the Metal recipe's own "next" list
  (parallel PNG bands, int8 weights and activations, step caching), cards 35, 50 and 51 (read-ahead, the resident
  server, resident weights), the maintainer's open H3-Image pull requests and h3-image-mlx (int8 W8A8, precomputed
  AdaLN, batched decoder tiles, streamed or resident weights) count only where a proposal clearly goes beyond them,
  said in its row.
- **No rejected lesson returns without new evidence, named.** See [past results that bind](#past-results-that-bind).
- **Precision.** Upstream takes no lossy weight or cache formats and no output-changing mode by default. An exact
  proposal keeps the latent and PNG byte-identical; a lossy one is opt-in, off by default with a byte-identical
  picture, and carries a picture-quality guard.
- **Each proposal has an A/B plan, a speed gate, a quality guard, an expected gain and a cost.**

### Shared A/B protocol

- **Held fixed:** checkpoint revision, prompt bytes, seed noise, the six sigmas and heads, decoder and precision
  outside the changed path.
- **Corpus:** the ideas page's 32 prompts (small text in Latin and Chinese, counting and spatial relations, faces and
  hands, thin structures, repeated textures, flat graphics, low-contrast scenes) x seeds {7, 42}, at 2048 x 2048 and
  4096 x 4096. Any tuning or calibration uses eight separate prompts.
- **Runs:** the same binary with the change off and on, alternated A-B-B-A in one resident process with no other GPU
  work; cold runs, every weight file evicted, reported apart from warm ones.
- **Recorded:** wall time to the PNG on disk, each stage's time, each step's GPU time, peak physical footprint.

### Speed gates

- **Local:** against card 34's step times, card 51's warm 2048 picture (12.70-12.77 s), card 35's cold pictures
  (61 s and 123 s) and card 34's 36-66 s PNG tail. A speed-up claim needs the lower 95% bound of the paired speed-up
  at or above the card's figure, timed through the PNG written.
- **External, every card:** h3-image-mlx's teapot prompt, seed 42, at 4096 x 4096: 356.84 s on an M5 Pro (343.6 s
  of steps, 8.37 s of decode). With the change on, our picture at that prompt, seed and size on this M5 Ultra, cold,
  takes at most 119 s (356.84 / 3; today 123 s). The machines differ, so this is a floor, not a like-for-like win;
  once card 3 reruns h3-image-mlx here, the bar is a third of that rerun's time at the same residency.

### Quality guards

- **Exact:** `cmp`-equal final latent and PNG at both sizes, five repeats, with the change off and on.
- **Lossy:** against the default picture at the same seed, LPIPS median at most 0.05 and worst at most 0.15, PSNR
  median at least 33 dB, at most one more OCR exact-match miss among the 16 typography pictures, no new error in
  object count or spatial relation, and the final latent's cosine at least 0.99, all on free-running six-step
  pictures. In addition, teacher-forced velocity cosine at least 0.999 at all six original heads, each head fed its
  own sigma's input from the default trajectory. Both halves are required; neither replaces the other, and a
  per-block check may stop a setting early but never passes one.

## Proposals, ranked

Ranked by expected gain times confidence. "Sweeps" counts the sweeps that proposed it independently; a dissent is a
sweep that turned the same idea down.

| Rank | Short name | Sweeps | Kind | Warm end to end, 2048 / 4096 | Cold end to end, 2048 / 4096 | Cost |
|---|---|---|---|---|---|---|
| 1 | `exact-png` | 3 | exact | none / x1.25-1.6 | none / x1.2-1.3 | 0.5 + 3-5 days |
| 2 | `block-budget` | 2 | lossy, opt-in | x1.4-1.65 / x1.25-1.35 | x1.05-1.1 / x1.2-1.3 | 1 + 5-7 days |
| 3 | `one-read` | 1 | exact | none | x1.5-1.75 / x1.2 | 0.5 + 3 days |
| 4 | `int8-attention` | 2 | lossy, opt-in | none / x1.05-1.12 | none / x1.05-1.1 | 0.5 + 3-5 days |
| 5 | `gemm-schedule` | 1, one dissent | exact | x1.05-1.3 / x1.05-1.13 | small | 1 + 3-5 days |

They work on different time: the PNG tail, the number of blocks run, the cold reads, attention's arithmetic and the
products' schedule, so cards 1-4 stack. With `exact-png` and `block-budget` across their expected ranges, warm 4096
falls from about 100 s to 36-60 s (x1.65-2.8); with `one-read` added, cold 2048 falls from 61 s to about 35 s. Warm
2048 moves only with `block-budget`.

### 1. `exact-png`: the same PNG bytes from every core, streamed behind the decoder

**Agreement: all three sweeps**, as a parallel deflate that rebuilds the sequential encoder's exact parse
(architecture), a same-stream speed-up of the existing compressor (history), and the PNG written band by band while
the decoder's tiles finish (outside).

- **Mechanism.** Zig's level-6 matcher hashes every position and stops its chain walk at the 32 KiB window, so the
  match found at a position depends only on the 32 KiB before it and the 258 bytes after. Split the filtered
  scanlines into one chunk per core; each worker hashes the 32 KiB before its chunk without emitting tokens, then
  parses speculatively. A join keeps worker k's tokens from the first position where worker k-1's true parse and
  worker k's both start a token; a chunk that never joins falls back to the serial parse. Blocks are cut every
  32,768 tokens as the serial writer does and encoded in parallel, each from its true starting bit offset: Zig
  0.17's compressor uses that offset to choose stored against Huffman blocks and to pad stored blocks, so offsets
  come from a prefix sum of the earlier blocks' exact bit lengths before any block is finalised, and the fuzz test's
  byte equality catches a wrong one. Adler-32 is combined across chunks. Bands start as the decoder places each tile
  row.
- **Beyond what exists.** The Metal recipe turns down parallel PNG bands because they change compressed bytes; this
  reproduces the existing bytes. Card 40 and h3-image-mlx work on the decoder's GPU tiles, not the host tail.
- **New evidence against the rejection.** The matcher's state is a function of the input window, so the speculative
  parse is the sequential parse, not an equivalent one. The card's equality test against `std.compress.flate` proves
  it or closes the card.
- **Expected gain.** At 4096 the tail falls from about 35 s warm (36-66 s one-shot) to about 2 s: warm about 100 to
  63-80 s, cold 123 to 95-100 s. 2048: under 2%.
- **Cost.** A half-day diagnostic, then 3-5 days. Host code only, in the shared picture writer, so other image
  families gain too.
- **A/B.** The diagnostic first: time `png.encode` alone, then blending and pixel conversion alone, on card 34's
  4096 picture with nothing else running. Then the shared protocol, and a fuzz test against the standard encoder on
  random, flat, repeated-pattern and noise buffers up to 8192 x 8192 and 1-64 chunks.
- **Speed gate.** PNG encode at most 2 s at 4096; the decoder stage after its weights at most 8.37 s (h3-image-mlx's
  whole decode); warm 4096 at most 66 s; 2048 not slower.
- **Quality guard.** Exact.
- **Risks.** If the diagnostic finds the tail was disk or CPU contention rather than deflate, the card closes with
  that split recorded; round 2 needs it either way.

### 2. `block-budget`: fewer blocks in steps 2-6, all six heads kept

**Agreement: history and outside**, as a fixed per-step block keep mask (history) and per-block forecasting across
steps (outside). Architecture did not propose it; its turned-down list rejects skipping steps or heads, which this
keeps.

- **Mechanism.** Every step runs at its own sigma with its own head; inside steps 2-6, a fixed set of whole blocks is
  not run. Two fills for a left-out block, as A/B arms: identity (the block's input passes through), and a first-order
  forecast of its output from the same block's last two steps. The keep mask per step and block comes from the
  calibration prompts and is frozen before the corpus. An explicit residual boundary handles the next block's
  deferred MLP add.
- **Beyond what exists.** Step caching (the recipe's "next" list, TeaCache, FBCache) skips whole steps, which the
  distilled heads forbid; this keeps every step and head and cuts depth. Feature caches (FORA, TaylorSeer, DeepCache)
  target 20-50-step schedules and nobody runs one on a six-step model with per-step heads, or on a Mac. No card, pull
  request or runner here skips blocks.
- **New evidence against the rejection.** The earlier turn-down of static block skipping rested on long schedules.
  Here no step is skipped, so each head still sees its own sigma; whether blocks are redundant within a step is the
  diagnostic's question, not assumed.
- **Expected gain.** If 40-55% of blocks in steps 2-6 can go: transformer x1.5-1.85; warm 2048 12.7 to 7.6-9.0 s
  (x1.4-1.65), warm 4096 about 100 to 73-80 s, cold 2048 61 to 56-57 s, cold 4096 123 to 96-103 s.
- **Cost.** A one-day diagnostic, then 5-7 days; the forecast arm holds two states per block (about 4.4 / 17.6 GB).
- **A/B.** The diagnostic first: per step and block, on the calibration prompts, the final latent's cosine when that
  block alone is left out, on free-running pictures; go only if leaving out a third of the blocks keeps cosine at
  0.99 or better. Then masks of increasing depth cut under the shared protocol; an all-keep mask is first proved
  byte-identical.
- **Speed gate.** Six steps at most 6.9 s at 2048 and 42 s at 4096 (card 34's / 1.6 and / 1.4), and the warm 2048
  picture at most 8.4 s (card 51's 12.70 s / 1.5), mask bookkeeping and forecasts counted.
- **Quality guard.** The complete lossy guard: free-running pictures and all six heads teacher-forced, per mask.
- **Risks.** A distilled model may need every block at every step; the diagnostic stops the card in a day. Small text
  and fine texture go first.

### 3. `one-read`: the cold picture's reads in compute order, one reader for all stages

**Proposed by architecture alone**, so its diagnostic decides.

- **Mechanism.** One pool of reader threads reads the prompt layers, the refiner, the transformer blocks, the step
  heads and the decoder in the order compute uses them. Each consumer waits per layer or block on a counter, not on
  its whole stage: step 1 runs under the transformer read's tail, and the decoder's weights are read during steps
  2-6. The resident server's first request uses the same schedule instead of loading everything first.
- **Beyond what exists.** Card 35's read-ahead (and #2999, which takes it upstream) runs per tool before that tool
  computes; its three reads are serial and none overlaps compute. h3-image-mlx streams blocks inside its transformer
  only. This crosses stage boundaries and gates compute per block.
- **New evidence against the rejection.** Past prefetch work was exact but competed for bandwidth. Here the prompt
  stage reads at 1.1 GB/s against the same disk's 3.8 GB/s, so the disk has room while the prompt runs; the
  diagnostic measures it.
- **Expected gain.** Cold 2048 61 to 35-40 s (x1.5-1.75), cold 4096 123 to about 100 s alone. Warm: none.
- **Cost.** A half-day diagnostic, then 3 days.
- **A/B.** The diagnostic first: disk throughput while card 35's prompt stage runs cold. Then card 35's protocol,
  every weight file evicted, three runs each, one-shot tools and the server's first request.
- **Speed gate.** Cold 2048 at most 40 s and cold 4096 at most 100 s (at most 119 s for the external bar), steps 2-6
  unchanged.
- **Quality guard.** Exact.
- **Risks.** If the prompt stage already saturates the disk, the card closes on its diagnostic.

### 4. `int8-attention`: attention's products on int8 tensor units

**Agreement: architecture and outside**, both as int8 query and key tiles with int32 sums and fp32 softmax;
architecture adds centring the keys first.

- **Mechanism.** Subtract each head's key mean (softmax ignores a per-query constant), quantise Q per 64-row tile and
  K per 64-key tile to int8, run QK^T as int8 `matmul2d` into int32, keep softmax and accumulation in fp32; P times V
  on int8 is a second phase.
- **Beyond what exists.** h3-image-mlx's W8A8 covers the linear layers only; SageAttention is CUDA-only; no runner
  does H3-Image's attention on Apple's integer tensor units. Pooled far keys cut the number of keys and this the cost
  of each, so they stack.
- **New evidence against the rejection.** The int8 negative here was weights dequantized before a bf16 product
  (ComfyUI, 1.7-1.9x slower). This keeps both operands int8 in the tensor op; the precondition measures whether that
  is faster.
- **Expected gain.** Attention x1.3-2 at 4096: step x1.09-1.2, warm 4096 x1.05-1.12, 2048 about 2%, so it stays off
  under 8,192 rows.
- **Cost.** A half-day microbenchmark, then 3-5 days.
- **A/B.** The precondition first: int8 `matmul2d` at the kernel's tiles at least 1.7x bf16's rate on this M5 Ultra.
  Then per-block attention output relative error on dumped activations, and the shared protocol at 4096.
- **Speed gate.** A 4096 step at most 8.6 s (QK only) or 7.9 s (QK and PV), against 9.80-9.86.
- **Quality guard.** The complete lossy guard: free-running pictures and all six heads teacher-forced; per-block
  attention output relative error at most 1e-2 only as an early stop.

### 5. `gemm-schedule`: H3-shaped product schedules, same sums

**Proposed by history.** **Dissent: architecture**, which turned down retiling without a roofline figure because the
products already run near 100 TFLOP/s. The outside sweep's variant, removing threadgroup staging, rests on a premise
the code refutes (the products already read from device memory).

- **Mechanism.** Grid swizzle and threadgroup grouping for the four projection classes at their real row counts,
  keeping each output's MMA chain and bf16 store.
- **Expected gain.** Transformer x1.1-1.35 at best; unknown until the bf16 tensor-unit peak is measured.
- **Cost.** A one-day roofline probe, then 3-5 days.
- **A/B.** Every production projection shape captured, the 24 text rows and ragged tails included; counters and
  per-projection GPU time before choosing a variant; then the shared protocol, cold compilation charged apart.
- **Speed gate.** At least 1.2x on the projection sum in production-shape replay, then six steps at most card 34's /
  1.1.
- **Quality guard.** Exact.
- **Why not a card.** Two sweeps doubt its headroom, and its best case is a few tenths. Card 4's microbenchmark
  records the bf16 peak; round 2 decides with it.

## Turned down

| Idea | Sweep | Reason |
|---|---|---|
| Resolution cascade, routed late attention, 512-pixel decoder tiles | all | Card 40's design |
| Pooled far-field keys; GPU and SME2 row split | all | Main's ideas page claims them |
| Read-ahead, resident server, resident weights, re-wiring per request | architecture, history | Cards 35, 50 and 51 |
| int8 / fp8 / fp4 / GGUF weights or activations in the linear layers | all | h3-image-mlx ships W8A8; lossy weight formats refused upstream; dequantized int8 is slower here |
| Lossless 12-bit bf16 weights decoded in the product | history | Earlier GPU decode was neither exact nor faster; these steps are compute-bound |
| Lossless checkpoint compression for cold reads | history | Unmeasured entropy and decode cost; `one-read` attacks the same time exactly |
| Parallel PNG bands, other filters or levels, libdeflate, zlib-ng | all | Change PNG bytes; `exact-png` gets the speed with the same bytes |
| GPU pixel conversion and blending; GPU Lanczos | history, outside | About a second at most, and blend order threatens bytes |
| Larger or more batched decoder tiles | history | Card 40's tiles; batches of 16 already run |
| Whole-step skipping, TeaCache, FBCache, merged steps, skipped or merged heads | all | Each step has its own distilled head; the recipe lists step caching; `block-budget` keeps every step |
| Re-distillation to fewer steps, MeanFlow, consistency models | history, outside | Needs new weights |
| Token merging, adaptive active rows | history, outside | Merged rows leave the learned patch layout; stale rows feed dense attention |
| Text or reference K/V reused across steps | all | Text rows take each step's timestep line, so K/V change; 0.6% of rows |
| Static FFN channel subset per step | architecture | x1.15 at warm 2048 at best, and no evidence a distilled model tolerates it |
| Unstructured activation sparsity in the FFN | architecture | Pays only when memory-bound; these products are compute-bound |
| SwiGLU and heads-scatter fusion into the products | architecture, history | At most 2-3% of a step; past exact fusions were slower |
| Removing threadgroup staging from the products | outside | They already read from device memory (`gemm_nax.metal`) |
| Split-K, fp16 accumulation in the products | architecture, history | Changes the sum order or loses precision at K = 14,336 |
| Larger query tiles in attention | architecture | Attention already runs faster than the products, not K/V-bound |
| Sliding-window, local-only or sparse attention | history, outside | Drops keys; pooled far keys keep them |
| Plain bf16 attention scores | outside | About 5% at 4096 with a weaker guard than `int8-attention` |
| Folding AdaLN into the weights | architecture | 231 GB of rewritten weights to save fused elementwise work |
| Graph capture, indirect replay, more queue depth, fewer barriers | architecture, history | One command buffer a step; `lap` barriers under 0.5%; past launch-gap ceilings near 1% |
| GPU-side Euler update | history | Removes small host copies between six long steps |
| Speeding the warm prompt encoder | architecture | 0.21 s, under 2% of a warm picture |
| Neural Engine or Core ML offload | architecture, outside | Cannot share the Metal stream; fp16 arithmetic; a rewrite |
| Overlapping the decoder with the steps; drafting the next step | architecture, outside | The decoder needs the final latent; nothing verifies a continuous draft cheaply |
| Spatial tiling of 4096 into 2048 windows | architecture | Breaks global coherence; pooled far keys do it better |
| Batching several pictures; multi-device sequence parallelism | history, outside | Throughput, or a second device; not one picture's time on one GPU |
| Conditioning and prompt caching | history, outside | Shipped elsewhere and near #2796; a fresh prompt cannot hit it |
| Exploiting the relative rotary lattice | architecture | Content differs at each offset, so no product is saved |
| Mixture-of-experts, Mamba-scan and KV-page wins from other models | history | Not the structure of this dense image transformer |

## Past results that bind

Our own speed rounds on other models set these limits. Each needs the named new evidence before it returns.

| Past result | Binds | New evidence needed |
|---|---|---|
| Launch-gap and graph-capture work measured about 1% of a span | Graph capture, launch fusion | A Metal trace with a larger idle share |
| Tile choices tuned on one model did not help another; generic "needs a bigger tile" claims were wrong once the code was read | `gemm-schedule` | Production-shape replay on H3-Image's own rows and a measured peak |
| Wider loads and deeper prefetch were exact but not reliably faster | `gemm-schedule`, `one-read` | Idle disk bandwidth during the prompt stage; a replay that beats the current products |
| Exact fusions of epilogues and norms were slower or inconclusive | Product fusions | A profile where the removed traffic outweighs register pressure |
| A decode-shaped attention win was slower on long prompts | Porting decode attention kernels | A kernel designed and measured for 4,000-16,000 query rows |
| One deterministic kernel did not make a whole model chunk-invariant | Calling any retiled kernel exact | Latent and PNG `cmp` on free-running pictures |
| Smaller or quantised weight files did not make inference faster | int8 / fp8 weight rows | Faster executed arithmetic, cold start included |
| Drafting that won in replay lost in served runs; a stronger-looking alternative lost to the default | `block-budget`, `int8-attention` | Free-running, end-to-end pictures, all overhead counted |
| Prompt-state reuse sped repeated conversations, not new prompts | Prompt and K/V caching | An exact reusable state with full input identity |
| A faster first output did not make the whole reply faster | Every card | Time to the PNG written, not a stage alone |
| Whole-model gates failed where every block passed (an earlier image model) | Every lossy card | Gates on free-running pictures, not block checks |
| Parallel PNG bands change compressed bytes | PNG work | The same bytes from a faster writer, as `exact-png` requires |

## Ready cards

Aimed at a large gain: cards 1 and 2 together target at least 1.5x on a warm 4096 picture against about 100 s, card 2
at least 1.5x on a warm 2048 picture against card 51's 12.7 s, and card 3 at least 1.5x on a cold 2048 picture
against card 35's 61 s. Order is the order to build. Each card starts with its diagnostic and closes as a measured
negative if it fails.

**1. MiniMax-H3-Image: the same PNG bytes from every core** (`exact-png`)

- A half-day diagnostic first: `png.encode`, blending and pixel conversion timed apart on card 34's 4096 picture
  with nothing else running; the split recorded in the recipe whatever it shows.
- A parallel writer that hashes each chunk's preceding 32 KiB, joins speculative parses and writes bands as the
  decoder places them; its PNG `cmp`-equal to `std.compress.flate`'s at 2048 and 4096 and on fuzzed buffers up to
  8192 x 8192 with 1-64 chunks.
- On the shared corpus on this M5 Ultra: PNG encode at most 2 s and warm 4096 picture at most 66 s in the resident
  server, 2048 not slower; latent and PNG byte-identical.
- The external bar: cold 4096 at h3-image-mlx's teapot prompt at most 119 s.

**2. MiniMax-H3-Image: fewer blocks in steps 2-6, every step and head kept** (`block-budget`)

- A one-day diagnostic first: per step and block, the final latent's cosine on free-running calibration pictures
  with that block left out; go only if a third of the blocks can go at 0.99 or better.
- `tf-h3-image blocks=MASK` with identity and forecast fills and an explicit residual boundary; without it, or with
  an all-keep mask, the final latent is byte-identical.
- On the shared corpus on this M5 Ultra: six steps at most 6.9 s at 2048 and 42 s at 4096, warm 2048 picture at
  most 8.4 s in the resident server; a table per mask and size in the recipe.
- The complete lossy guard, alone and stacked with card 1: free-running pictures, and teacher-forced velocity cosine
  at least 0.999 at all six original heads, each fed its own sigma's input from the default trajectory.

**3. MiniMax-H3-Image: one reader for every stage's weights, in compute order** (`one-read`)

- A half-day diagnostic first: disk throughput during card 35's cold prompt stage; stop if it is already near the
  disk's 3.8 GB/s.
- Reads gated per layer and block, step 1 running under the transformer read's tail, the decoder read during steps
  2-6, in the one-shot tools and the server's first request.
- Every weight file evicted before each run: cold 2048 at most 40 s and cold 4096 at most 100 s on this M5 Ultra;
  latent and PNG byte-identical to card 35's build.
- h3-image-mlx rerun on this M5 Ultra with its teapot prompt, seed 42, at 4096, cold and warm; its time replaces the
  356.84 s in every later card's external bar.

**4. MiniMax-H3-Image: attention's products on int8 tensor units, opt-in** (`int8-attention`)

- A half-day microbenchmark first: int8 `matmul2d` at the attention kernel's tiles at least 1.7x bf16's rate on this
  M5 Ultra, with the bf16 peak recorded for round 2; else close.
- `tf-h3-image attention=int8` with centred keys, from 8,192 rows up; without it, the final latent is
  byte-identical.
- A 4096 step at most 8.6 s (QK) or 7.9 s (QK and PV); per-block attention output relative error at most 1e-2 as an
  early stop; the complete lossy guard at 4096, alone and stacked with card 2: free-running pictures, and
  teacher-forced velocity cosine at least 0.999 at all six original heads, each fed its own sigma's input from the
  default trajectory.

## What round 2 starts from

Round 2 reads these measurements, not this report's estimates:

- **The warm 4096 stage split** in the resident server, and card 1's diagnostic split of the PNG tail. Today's 100 s
  is one diagnostic run; every 4096 gain above rests on it.
- **This round's diagnostics:** the per-block leave-one-out cosine (card 2), the disk's idle bandwidth during the
  prompt stage (card 3), int8 and bf16 tensor-unit peaks (card 4). Each is a result whether it says go or stop; the
  peak decides `gemm-schedule`.
- **Each build card's A/B table:** speed and every guard metric per setting. A setting that fails its guard enters
  the binding table above with its numbers; one that passes sets the new baseline for the next card's gate.
- **Card 40's cascade and main's `far=pool` results**, so round 2 knows which lossy cards stack with cards 2 and 4.
- **The same-machine h3-image-mlx run** (card 3), which replaces the cross-machine 356.84 s as the external gate.

Round 2 repeats the three sweeps from those numbers, on three model families running at the same time in separate
checkouts, with the binding table extended by round 1's own negatives.
