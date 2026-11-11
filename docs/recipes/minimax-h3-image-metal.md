# MiniMax-H3-Image on a Mac: what nobody does yet

MiniMax-H3-Image's transformer is MiniMax H3's without audio: 50 single-stream blocks of 5,376, 56 heads of 128, a
SwiGLU of 14,336, about 17.3 B bf16 parameters. One row covers 32 x 32 pixels (the VAE's 16, then a 2 x 2 patch), so a
2048 x 2048 canvas is 4,096 picture rows and 4096 x 4096 is 16,384. It is distilled to six Euler steps with no
guidance, each with its own output head. Its text encoder is Qwen3-VL's text model cut at layer 50; its decoder is a
24-block ViT run over overlapping 256-pixel tiles.

This page lists what every other way of running it already does for speed, the ideas none of them ship for it on
Apple silicon, and the one this fork has started. The maintainer's Metal port is on `upstream/lead/h3-image*`; its own
recipe documents the tools and their gates.

## Cost per step

Per block, N rows: projections 2 N x 346.8 M FLOP, attention about 4 N^2 x 5,376. Over 50 blocks:

| Canvas | Rows | One step | Attention's share |
|---|---|---|---|
| 2048 x 2048 | 4,096 | 160 TFLOP | 11% |
| 4096 x 4096 | 16,384 | 857 TFLOP | 34% |

The decoder is about 1.6 s at 2048 x 2048 on an M3 Ultra (121 tiles of 13.4 ms) and about 5.9 s at 4096 x 4096 (441
tiles): small beside six transformer steps at that size.

## What everyone already does

- **The maintainer's TensorFold port.** bf16 weights read in place, M5 tensor-unit GEMMs with fp32 sums, dense tiled
  attention, a float64-gated kernel contract. Five separate tools: prompt, vision, edit encoder, transformer, decoder.
- **vLLM-Omni** ([recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image)). Sequence and tensor parallel layouts,
  layer offload for small GPUs, CUDA graphs, a request-level conditioning cache. CUDA only.
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

- **Gain.** A half-side step costs 1/5.35 of a 4096 step (attention falls 16x). Six steps with m = 2, 3, 4 low:
  1.37x, 1.69x, 2.18x faster at 4096; 1.35x, 1.63x, 2.06x at 2048 (half side 1024).
- **Risk.** Layout is decided at a quarter of the rows. Detail forms in the last, full-size steps, but a distilled
  trajectory may not forgive the jump: softness, or seams at 2 x 2 blocks.
- **Proof.** Twenty seed-locked prompts at 4096 x 4096 on one M5, picked before the run: median wall time against the
  uncascaded run and LPIPS of the decoded pictures against it, per m. Gate: m = 3 at least 1.6x faster with LPIPS at
  most 0.08.
- **Needs weights?** Only to measure. The lift is started here (below).

### 2. Routed attention in the last full-size steps

At 4096 x 4096 attention is a third of a step. This fork already has FastH3's routed attention on Metal: each
64-row tile attends the prefix and its top tiles by pooled score (`tf-h3-dit`). H3-Image was not trained with it and
has no gate for the pooled branch, so nobody runs it there, but its last steps only sharpen a settled picture, where
attention is most local. Route the target tiles in steps 4 and 5 only, without the pooled branch, keep the text and
references dense, and pick `keep` by measurement.

- **Gain.** At keep = 20%, a 4096 step's attention falls to about a fifth: 27% off each routed step, 9% off the
  run alone, about 15% stacked on the cascade (whose last steps are the full-size ones). These are FLOP bounds: the
  dense prefix and the routing pass come off them.
- **Risk.** Untrained routing can drop a far tile that mattered: repeated textures and symmetric objects are the
  first to break. Text and reference rows stay dense.
- **Proof.** `keep` chosen on ten prompts apart from the twenty, including edits; then on the twenty at 4096, LPIPS
  against dense steps with the same cascade at most 0.03, and the routed steps' GPU time against those dense ones.

### 3. The decoder in 512-pixel tiles on the tensor units

The decoder's products run without tensor units (about 12 TFLOP/s on an M3 Ultra, by its 13.4 ms a tile), and its
256-pixel tiles overlap: 441 tiles at 4096 x 4096, 1.78x the picture's area. h3-image-mlx already runs its decoder
through MLX's GEMMs on an M5 (8.37 s at 4096), so tensor units alone are not new; 512-pixel tiles are (overlap 1.31x),
at four times each tile's attention per token.

- **Gain.** 5.9 s to about 1.5 s at 4096 x 4096 [INFERENCE: the M5 tensor-unit rate on 1,024-wide products].
- **Risk.** Tensor-unit GEMMs change sum order: within the decoder's float64 gate, not bit-equal. A larger tile changes
  what each token attends to, so it is checked against the reference decode, not assumed.
- **Proof.** The decoder's selftest gates unchanged; on five fixed latents, PSNR at least 45 dB against the reference
  decoder at 4096.

### Considered, not ranked

- **Caching the prompt's rows, or keeping its encoder apart from the transformer.** SGLang and vLLM-Omni cache
  conditioning; h3-image-mlx and the maintainer's tools run the encoder and the transformer as separate processes.
- **Reusing an edit reference's rows across steps.** Not exact: its hidden states depend on the moving target rows
  through attention. A lossy cache stays opt-in under this repository's precision rules.
- **Streaming bf16 blocks from the SSD on 24-36 GB Macs.** A block is 0.69 GB; at 4096 a block's compute is fast
  enough that 5-7 GB/s of SSD cannot keep up, so it works but is not fast. Community MLX ports of the video model
  stream this way [INFERENCE: from their READMEs, not run here].

## Follow-up cards, in order

1. **H3-Image: the resolution cascade in `tf-h3-image` (`cascade=M`).**
   - Steps 0..M-1 on the half grid's tiles and positions, then `cascade.lift`, then steps M..5 at full size.
   - Without `cascade` the run is byte-identical to today's.
   - At 4096 x 4096 on an M5, twenty seed-locked prompts: M = 3 at least 1.6x faster with LPIPS at most 0.08 against
     the uncascaded run; the table per M in this recipe.
2. **H3-Image: routed target attention in the last two steps (`route=KEEP`).**
   - The existing routed attention kernels, with no pooled branch; text and reference rows dense.
   - Without `route` the run is byte-identical; at 4096 on an M5, LPIPS against dense steps with the same `cascade`
     at most 0.03 and each routed step at least 20% faster.
3. **H3-Image decoder on the M5's tensor units in 512-pixel tiles.**
   - Today's 4096 x 4096 decode time recorded first on the same M5; the selftest's float64 gates hold and the decode
     is at least 3x faster than that record.
4. **H3-Image at 4096 x 4096 at least 3x faster than h3-image-mlx**, cards 1 to 3 together: its teapot prompt, seed
   42, text encoding through PNG, both on the same M5 (356.84 s on an M5 Pro today, so under 120 s there).

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
has been made with it (card 1).
