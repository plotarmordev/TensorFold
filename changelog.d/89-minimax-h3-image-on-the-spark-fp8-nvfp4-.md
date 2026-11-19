# MiniMax-H3-Image on CUDA: block-scaled int8 on every projection
- GB10's block-scaled FP8 (cuBLASLt MXFP8, e4m3 at a UE8M0 scale every 32 values) was measured and reverted first: it
  missed episode 1's gates (latent cosine 0.9664 against 0.99, the picture 11.53 apart from its Mac PNG against 4.1)
  and was slower than int8, 19.5 s warm against 18.9 s. int8 keeps three mantissa bits of headroom e4m3 does not.
- Every transformer projection — q k v gate, attention out, mlp in and mlp out — now runs as `h3_gemm_int8`, an
  int8 `mma.m16n8k32` tile that holds a 128-value chunk's exact int32 sum and folds it into the fp32 accumulator at
  that chunk's activation and weight scales, then writes bf16 straight from the tile. `h3_quantize` scales each
  128-wide chunk by its own absmax, so `out` and `fc2` leave bf16 without the per-row scale's distance from episode
  1, and there is no int32 round trip through a dequantize pass.
