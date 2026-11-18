# MiniMax-H3-Image: how mmh3 gets 17.6 s on a Spark, and what we copy
- `docs/mmh3-h3-image-study.md` reads mmh3 at `9c0652f` — its code, its docs and its pull-request write-ups — and
  lists what it does differently: INT8 ConvRot linears with per-row activation scales after a Hadamard rotation, a
  FlashAttention-2 dense kernel, INT8/FP8 attention, Sol-Attn's routed 64-token blocks, per-block fused
  preprocessing, GPU-side step heads, PDMD few-step LoRAs and half-canvas drafts, a 2048² minimum canvas, FP16 VAE
  products with every tile in one batch, kept prompt states, cuBLASLt and INT8 tiling caches, threaded in-place reads,
  and no CUDA graphs anywhere.
- The same prompt, seed 42, six steps and 2048² measured on TOP: mmh3 18.6 s (17.6 s on BOTTOM's GB10) against our
  warm 31.4 s, and mmh3's warm server is 9.5 s. Our step is 4.08 s against its 1.4 s.
- Ranked by saving against our 31.4 s: INT8 ConvRot linears −16.1 s, FP16 VAE products and one tile batch −4.6 s,
  fewer steps −4.2 s, kept prompt states −1.1 s, INT8/FP8 attention −0.6 s at 2048² and −9 s at 4096², GPU-side
  step heads 0.1–0.5 s, algorithm caches 4–7 s off a cold first step. Together they put a warm 2048² picture at
  9.5 s, and about 5.3 s with three steps, if the picture gates hold.
- Card 83's INT8 round is reshaped from that list: INT8 ConvRot DiT linears and the FP16 VAE first, measured beside
  mmh3's own run in one TOP window, with 9.5 s as the number to beat; fewer steps after.
