# MiniMax-H3-Image on CUDA: own int8 tile removed, all-four cuBLASLt int8 measured and reverted

- The card's own-tile `h3_gemm_int8` is gone: three geometries measured 0.74–0.79× of cuBLASLt's
  ~152 TOPS on the picture's shapes (recipe `docs/recipes/minimax-h3-cuda.md`), under the card's
  stop-before-pictures rule. The kernel, its `h3_quantize_chunks` activation path and their
  contracts were removed so main carries no unused kernel.
- The rescope — `out` and `fc2` on the same cuBLASLt int8 path as `qkvc` and `fc1` — measured
  0.6 s faster warm at 2048² (13.7 vs 14.3 s, no CUDA graph) but failed the episode-1 picture
  gates (28.15 dB, mean 4.23, cosine 0.9945 against 28.2 dB / 4.1 / 0.99): per-row scales are
  the only fold cuBLASLt's int32 sum supports, and the finer scales want an own-mma product.
  `out` and `fc2` stay in bf16; `qkvc` and `fc1` keep their int8 default.
