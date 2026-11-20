# MiniMax-H3-Image on the Spark: own int8 TMA product on `qkvc`/`fc1`

mmh3's TMA + persistent-grid int8 GEMM (mma.sync.m16n8k32.s8, two-stage mbarrier ring, one CTA per SM
walking `group_m` bands) ported into the H3 CUDA path, replacing cuBLASLt int8 + h3_dequantize +
h3_swiglu_int8 + h3_heads_i8. The product scales its exact int32 sums to bf16 in registers — the
same products and roundings the deleted kernels computed — so latents stay bit-identical to the
path it replaces; `h3_swiglu_rows` interleaves fc1's gate/up weight rows pairwise for the fused
SwiGLU epilogue. `out`/`fc2` stay bf16 (per-row scales still fail the picture gates; receipts under
cards 129/139 in `docs/recipes/minimax-h3-cuda.md`).
