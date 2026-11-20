- MiniMax-H3-Image's Metal picture path runs the norm, quantize and activation passes fused, as card 128 did on the
  Spark: `h3_norm_quant` folds the gated residual add and the modulated RMSNorm straight into the int8 rows `xq`/`xs`
  (the bf16 normed rows are never written — `h3_gemm_32x64_i8` is their only reader), and `h3_gemm_32x64_i8_act`
  folds `silu(gate)·value` into fc1's int8 product epilogue, sweeping K once over the gate rows and once over the
  value rows and writing `act` directly — `h3_swiglu` and the `wide` rows drop out of the block. Two launches fewer
  a block on the released layout (10 → 8), bit-identical to the three launches they replace; the clip's bf16 path is
  unchanged.
