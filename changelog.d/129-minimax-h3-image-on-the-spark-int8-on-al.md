# MiniMax-H3-Image on CUDA: all four projections int8 on a rebuilt own tile
- Every DiT projection (q k v gate, attention out, mlp in/out) runs int8 by default: weights quantized
  per-row (one fp32 scale a row), activations rotated in 128-wide chunks with one fp32 scale a chunk —
  the same recipe card 89 proved accurate (29.51 dB / 0.9959 cosine), replacing main's cuBLASLt mix
  (qkvc + fc1 int8, out + fc2 bf16).
- `h3_gemm_int8` was rebuilt with per-chunk activation scales folded in registers (2 fma/elt, the mma's
  12582912.0f bias trick) and per-row weight scales applied once in the epilogue — vs the reverted tile's
  per-chunk scale smem staging and ~5 ops/elt fold. bf16 is written straight from the mma accumulators:
  `h3_dequantize` and `h3_swiglu_int8` (the int32 round trip) are gone; `mlpIn` feeds the same bf16
  `h3_swiglu` the unquantized path uses.
- New kernel `h3_quantize_chunks` produces the activation-side per-chunk scales; `h3_quantize` keeps the
  weight-side per-row scales.
