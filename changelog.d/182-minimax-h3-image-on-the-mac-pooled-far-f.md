# MiniMax-H3-Image on the Mac: pooled far-field keys at 4096, opt-in (`--far-pool`)

- `tf-h3-image` and `tf-h3-image-server` take `--far-pool`: at 4096-wide canvases of 8,192 target rows and more,
  the last two steps' target query tiles attend the prefix, a band of 16 full-resolution picture rows around them,
  and the rest of the picture as 2 x 2 mean-pooled keys and values with ln 4 on their logits (`h3_far_pool`,
  `picture.far`, `picture.far_from_step`): 89 key tiles in place of 257 at 4096 x 4096.
- On an M5 Ultra, warm (seed 7) 4096's six steps go from 55-59 s to 49-50 s; 4096 x 4096 pictures pass card 80's
  gates (cosine 0.9991, PSNR 31.0-31.3 dB, mean 3.6-3.8). Pooled on every step it was 40 s but laid the picture out
  differently.
- Off by default: `h3_attention`'s far branch compiles only with the flag, and default pictures are byte-identical
  to main's at 2048 and 4096. 2048 stays dense with the flag.
- Tests: every target key read once, full or pooled (`test-h3-geometry`); attention with pooled far keys matches dense
  attention to one bf16 step when each 2 x 2 block's keys and values are equal, and routed steps keep each head's
  chosen tiles (`test-h3-gemm`); `h3_far_pool` against float64 through the kernel contract's gates.
