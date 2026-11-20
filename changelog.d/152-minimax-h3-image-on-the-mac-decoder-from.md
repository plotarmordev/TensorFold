# MiniMax-H3-Image: the Mac decoder on the tensor units

- `tf-h3-image-decoder`'s products run on `nax.h`'s fragments (the M5's tensor units, simdgroup matrices elsewhere), with bf16 operands and float32 sums at the same rounding points. `h3d_swiglu` runs four values per thread, and `h3d_attention` loads k and v as vectors.
- Warm 2048x2048 decode on an M5 Ultra: 121 tiles now take 0.56 s on the GPU, down from 1.08 s, and the server's decoder stage 0.80-0.85 s, down from 1.35-1.37 s. Warm totals are 11.28-11.30 s on unseen prompts, down from 11.59-11.61 s; fast mode (`--draft-steps 3`) is 9.42-9.51 s, down from 10.00-10.53 s. Latents are bit-identical to main's. Receipts are in `docs/recipes/minimax-h3-image-metal.md`.
