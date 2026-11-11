# MiniMax-H3-Image on a Mac: what nobody does yet

MiniMax-H3-Image's transformer is MiniMax H3's without audio: 50 single-stream blocks of 5,376 with attention 7,168
wide (56 heads of 128), a SwiGLU of 14,336. The full DiT is 33.1 B bf16 parameters: 19.3 B of per-row projections,
13.0 B of per-block AdaLN tables that depend only on the step, and a 0.8 B text refiner. One row covers 32 x 32
pixels (the VAE's 16, then a 2 x 2 patch), so a 2048 x 2048 canvas is 4,096 picture rows and 4096 x 4096 is 16,384.
It is distilled to six Euler steps with no guidance, each with its own output head. Its text encoder is Qwen3-VL's
text model cut at layer 50; its decoder is a 24-block ViT run over overlapping 256-pixel tiles.

This page lists what every other way of running it already does for speed, the ideas none of them ship for it on
Apple silicon, and the one this fork has started. The ranking is FAB's (`claude-fable-5-2`), run blind on the facts
above and the list below. The maintainer's Metal port is on `upstream/lead/h3-image*`; its own recipe,
[minimax-h3-image-metal.md](minimax-h3-image-metal.md), documents the tools and their gates.

## Cost per step

Per block, N rows: projections 2 N x 385.35 M FLOP (QKV 5,376 x 21,504, output 7,168 x 5,376, SwiGLU 5,376 x 28,672
and 14,336 x 5,376), attention 4 N^2 x 7,168, from `transformer/config.json` and the safetensors shapes. Over 50
blocks, without the text rows:

| Canvas | Rows | One step | Attention's share |
|---|---|---|---|
| 2048 x 2048 | 4,096 | 182 TFLOP | 13% |
| 4096 x 4096 | 16,384 | 1,016 TFLOP | 38% |

The decoder is about 1.6 s at 2048 x 2048 on an M3 Ultra (121 tiles of 13.4 ms) and about 5.9 s at 4096 x 4096 (441
tiles): small beside six transformer steps at that size.

## What everyone already does

- **The maintainer's TensorFold port.** bf16 weights read in place, M5 tensor-unit GEMMs with fp32 sums, dense tiled
  attention, a float64-gated kernel contract. Five separate tools: prompt, vision, edit encoder, transformer, decoder.
- **vLLM-Omni** ([recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image)). Sequence and tensor parallel layouts,
  layer offload for small GPUs, CUDA graphs, a request-level conditioning cache. NVIDIA and AMD Instinct GPUs.
- **SGLang diffusion** ([cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-Image)). AdaLN tables
  precomputed and their weights freed, CUDA graphs, regional `torch.compile`, fused QK-norm and rotary, batched decoder
  tiles, a conditioning cache, layerwise offload, optional FP8 weights. CUDA only.
- **diffusers.** The reference pipeline in plain PyTorch; on a Mac, MPS with no Apple-specific path.
- **ComfyUI** ([Comfy-Org repack](https://huggingface.co/Comfy-Org/MiniMax-H3)). int8 text encoders, fp8, w4a8 and
  nvfp4 transformer repacks, GGUF for the video model, and a two-step re-distillation that needs new weights.
- **H3 video engines** (this fork's `tf-h3-dit`, drowzeys' Python H3 family). FastH3's top-20% tile attention, held
  keyframe rows, layer-streamed weights for small Macs in community MLX ports.
- **h3-image-mlx** ([repo](https://github.com/Anionex/h3-image-mlx)). The fastest published Mac run: Comfy-Org's
  int8 weights through W8A8 Metal tensor ops, the prompt encoder, transformer and decoder in separate processes,
  resident or block-streamed weights, AdaLN tables precomputed, batched decoder tiles. 4096 x 4096 in 356.84 s on an
  M5 Pro (343.6 s of transformer steps, 8.37 s of decode).

None of them change the canvas inside the six steps.

## Ideas nobody ships, ranked

### 1. A resolution cascade inside the six steps

The first m steps run on a half-side canvas, the rest on the full one. The positions allow it: a picture block's
positions are normalised by its own area, so a half canvas spans the same [0, 32) coordinates in steps twice as wide,
and each step's head is chosen by step, not size. What the positions cannot hide is the coarser row density, which is
what the proof measures. A 2048 canvas is the model's own default size, so a 4096 run's early steps sit at a size the
model knows.

- **Gain.** A half-side step costs 1/5.59 of a 4096 step (attention falls 16x). Six steps with m = 2, 3, 4 low:
  1.38x, 1.70x, 2.21x faster at 4096; 1.35x, 1.63x, 2.07x at 2048 (half side 1024). Lifting twice, 2 steps at a
  quarter side, 2 at half, 2 at full, gives 2.46x at 4096.
- **Risk.** Layout is decided at a quarter of the rows. The late heads were distilled expecting fine detail already
  present at their sigma: soft micro-texture, broken small text, or seams at 2 x 2 blocks.
- **Proof.** 32 prompts picked before the run, two seeds each, at 4096 x 4096 on one M5: 8 each of small typography,
  faces and hands, fine texture, and architecture. Against the uncascaded bf16 run: top-octave radial spectrum energy
  at least 0.90 of it, OCR exact matches on the typography set at most 1 of 16 fewer, LPIPS at most 0.08, and
  transformer wall time at most 0.50x for m = 4.
- **Needs weights?** Only to measure. The lift is started here (below).

### 2. Pooled far-field keys at 4096

Each 64-row query tile attends full-resolution keys in a 32 x 32-row window around it; picture keys and values
beyond it are 2 x 2 mean-pooled, with ln 4 added to each pooled logit so one key stands for its four. Text and
reference rows stay dense. Unlike FastH3's routing nothing is dropped and there is no routing pass, and it is exact
when the four keys are equal. PixArt-Sigma trains a similar key compression in; nobody applies one, training-free, to
H3-Image.

- **Gain.** At 16,384 rows each query sees 1,024 + 15,360 / 4 = 4,864 keys: attention falls 3.37x (385 to 114
  TFLOP a step), a step 1.36x. Stacked on the m = 4 cascade's two full steps: 2,219 TFLOP against 6,097, 2.75x. At
  2048 attention is 13% of a step, so it stays off below 8,192 rows.
- **Risk.** Exact long-range copies blur: repeated patterns, a glyph matching another far away. QK-norm bounds the
  logits, so the error is bounded by the spread of keys inside a pooled block.
- **Proof.** The 32 prompts, same trajectory as dense: per-block attention output relative L2 at most 1e-2 on dumped
  activations, PSNR at least 35 dB and LPIPS at most 0.05 on the decoded pictures, attention kernel time at most
  0.35x dense at 16,384 rows.

### 3. Splitting each product's rows between the GPU and the CPU's SME2 units

The linear layers are 62-87% of a step and row-independent. Unified memory lets the CPU's matrix units read the same
bf16 weights and activations with no copy, so each product's rows split in the measured throughput ratio. Every
other runner leaves the CPU idle.

- **Gain.** 1.05-1.07x at 4096 if SME2 sustains 1.5-2 TFLOP/s beside about 18 on the GPU (h3-image-mlx's rate)
  [INFERENCE: unmeasured].
- **Risk.** None beyond fp32 summation order; shared memory bandwidth and heat may eat the gain.
- **Proof.** A 5,376 x 14,336 product at 16,384 rows split against the GPU alone, at least 1.07x; PSNR at least 50
  dB against GPU-only pictures; no slowdown over three back-to-back 4096 runs.

### Considered, not ranked

- **FastH3's routed attention in the last steps.** The video engines already ship it, and it drops tiles where
  pooled far-field keys keep them.
- **The decoder in larger tiles on the tensor units.** About 1.7% of a 4096 run.
- **Caching the prompt's rows, or keeping its encoder apart from the transformer.** SGLang and vLLM-Omni cache
  conditioning; h3-image-mlx and the maintainer's tools run the encoder and the transformer as separate processes.
- **Reusing a step's residuals or an edit reference's rows across steps.** Six steps, each with its own head, and
  the reference's hidden states move with the target rows: not exact, so opt-in at most.
- **Merging similar picture rows.** Merged rows fall outside the 2 x 2 patch distribution the model learned.
- **Streaming bf16 blocks from the SSD on 24-36 GB Macs.** A block is 0.69 GB; at 4096 a block's compute is fast
  enough that 5-7 GB/s of SSD cannot keep up, so it works but is not fast. Community MLX ports of the video model
  stream this way [INFERENCE: from their READMEs, not run here].

## Follow-up cards, in order

The bar is h3-image-mlx's 356.84 s at 4096 x 4096 on an M5 Pro, at int8; these cards beat it at bf16.

1. **H3-Image: the dense bf16 baseline at 4096 x 4096 on an M5 Pro.**
   - Wall time per stage published, and the 32 prompts' two-seed reference pictures stored outside the repository.
2. **H3-Image: the resolution cascade in `tf-h3-image` (`cascade=M`, one or two lifts).**
   - Low steps on the half (or quarter) grid's tiles and positions, then `cascade.lift`, then the rest at full size.
   - Without `cascade` the run is byte-identical to today's.
   - m = 4 passes idea 1's gates with transformer time at most 0.50x card 1's; the table per schedule in this recipe.
3. **H3-Image: pooled far-field keys at 16,384 rows and above (`far=pool`).**
   - A kernel test that equal keys give dense attention's output; off below 8,192 rows; idea 2's gates hold.
4. **H3-Image: GPU and SME2 row split for the linear layers.**
   - At least 1.07x on idea 3's product with PSNR at least 50 dB.
5. **H3-Image at 4096 x 4096 in at most 180 s on an M5 Pro, at least 1.98x faster than h3-image-mlx**, cards 2 to 4
   together, every quality gate green; at 2048 x 2048 faster than the best published Mac figure at that size.

## The cascade's lift (started here)

`zig/src/families/h3/cascade.zig` is the host side of idea 1, on latents [channels, rows, columns] in float32:

- `half(full)`: the low grid. Both grids keep even sides, so canvas sides are multiples of 64 pixels.
- `pool(full_noise, full, out)`: the half run's starting noise from the seed's full draw: each 2 x 2 block's sum over
  2. Unit noise stays unit noise, and both canvases come from one seed.
- `lift(x, v, from, to, full_noise, full, out)`: the full canvas at sigma `to` from the last half step's input `x` at
  sigma `from` and its velocity `v`, folding that step's Euler move into the lift. The clean estimate x + from v is
  upsampled bilinearly (half-cell centres, edges clamped). The noise is the seed's full draw with each 2 x 2 block
  shifted so it pools to the step's own noise estimate x - (1 - from) v. For an i.i.d. standard Gaussian seed and an
  estimate that depends on the seed only through its pools, that is the exact conditional draw (up to float32
  rounding): low frequencies from the half run, fine ones from the seed. `out` may not overlap `x`, `v` or the noise.

`zig build test-h3-geometry -Dh3-filter=cascade` runs its tests with no GPU or weights: the pool's sums and unit
variance, the bilinear taps on a ramp, the exact flow identity on two planes, the shifted noise's pools and kept
detail, and refusals of odd grids, wrong lengths and out-of-range or NaN sigmas.

Not done: the cascade is not wired into `tf-h3-image`, which is on the maintainer's unmerged branches, and no picture
has been made with it (card 2).
