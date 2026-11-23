# MiniMax-H3-Image on CUDA: the int8 TMA product takes a partial last column tile

- `h3_gemm_int8_tma` and `h3_swiglu_int8_tma` (`h3_int8_tma.cuh`) floored N to whole 128-column tiles, so at
  16x392x128 columns 384-391 were never written and read back as NaN. The grid now rounds N up: TMA zero-fills
  the edge tile's weight rows, its lanes past N clamp their scale loads and skip the store. `cuda_int8.zig` takes
  any N divisible by 8 (one 16-byte store a lane). `h3_swiglu_int8_tma` holds 225 registers (219 before).
- The contract serves 16x392x128, 300x1352x256 and SwiGLU's 300x776x256 and sweeps N in steps of 8, replacing
  the control that planted the unwritten columns. The picture's projections are whole tiles: their 2048²
  latents and PNGs are main's bytes, warm wall within noise of main's (11.5 / 11.8 s against 11.4 / 12.0 s,
  two jobs).
