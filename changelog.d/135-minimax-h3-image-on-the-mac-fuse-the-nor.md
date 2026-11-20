# MiniMax-H3-Image on the Mac: fuse the norm, quantize and activation passes

- MiniMax-H3-Image's Metal picture path fuses the norm and quantize passes, as card 128 did on the Spark:
  `h3_norm_quant` folds the gated residual add and the modulated RMSNorm straight into the int8 rows `xq`/`xs`,
  a scale a 256-value K block a row in the gemm's own layout (the bf16 normed rows are never written —
  `h3_gemm_32x64_i8` is their only reader). One launch fewer a block on the released layout, bit-identical to
  the two launches it replaces; the clip's bf16 path is unchanged. An activation-in-epilogue variant of fc1's
  int8 product measured slower than `h3_gemm_32x64_i8` + `h3_swiglu` (~1.63 vs ~1.58 s of GPU a step at 2048)
  and was dropped in review. `tf-h3-image-server`'s argument parse (its positionals no longer hold argv[0]) is
  fixed so the binary runs again.
