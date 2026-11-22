# MiniMax-H3-Image on the Mac: pooled far-field keys at 4096, opt-in (`--far-pool`)

- `tf-h3-image` and `tf-h3-image-server` take `--far-pool`: at 4096-wide canvases of 8,192 target rows and more,
  each target query tile attends the prefix, a band of 16 full-resolution picture rows around it, and the rest of the
  picture as 2 x 2 mean-pooled keys and values with ln 4 on their logits (`h3_far_pool`, `picture.far`). That is 89
  key tiles in place of 257 at 4096 x 4096. Off by default: `h3_attention`'s far branch compiles only with the flag.
- Tests: every target key read once, full or pooled (`test-h3-geometry`); attention with pooled far keys matches dense
  attention to one bf16 step when each 2 x 2 block's keys and values are equal (`test-h3-gemm`).
- Warm 2048/4096 times with and without the flag, and the picture gates, are not measured yet: they need a Mac GPU
  window longer than the two-minute cap. The A/B script is in `docs/recipes/minimax-h3-image-metal.md`.
