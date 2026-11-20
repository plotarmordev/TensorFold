# MiniMax-H3-Image on CUDA: all four DiT projections on cuBLASLt int8

- `out` and `fc2` join `qkvc` and `fc1` on the existing int8 path (rotated int8 weights and rows
  at per-row scales, cuBLASLt's exact int32 sums, `h3_dequantize`/`h3_swiglu_int8` back to bf16):
  286 TFLOP a picture of int8 work the bf16 default ran at ~84 TFLOPS, one third of the projections'
  kernel time. `--bf16` still runs everything in bf16.
- The card's own-tile `h3_gemm_int8` is gone: three geometries measured 0.74–0.79× of cuBLASLt's
  ~152 TOPS on the picture's shapes (recipe `docs/recipes/minimax-h3-cuda.md`), under the card's
  stop-before-pictures rule. The kernel, its `h3_quantize_chunks` activation path and their
  contracts were removed so main carries no unused kernel.
