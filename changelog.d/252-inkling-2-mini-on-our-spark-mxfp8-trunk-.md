# Inkling 2 Mini on CUDA: the trunk (attention projections and unembedding) quantised at load behind TF_INKLING2_TRUNK

- `TF_INKLING2_TRUNK` = `mxfp8` or `nvfp4` (default `bf16`) packs Mini's fused attention projections, `wo` and the
  unembedding into the lane matmul's MXFP8 (e4m3 codes, a power-of-two scale byte a 32 inputs) or NVFP4 (e2m1 codes,
  an e4m3 scale a 16, a matrix global) rows as `mini_weights.zig` loads them, instead of bf16. Every trunk product
  halves or quarters its weight bytes a decode step reads.
- `core/quant/mxfp8.zig` gains `encodeRow` (each 32-input group's least power of two under e4m3's 448, then nearest
  e4m3) and `core/quant/nvfp4.zig` `encodeRow` (the matrix-scale block search windowed to ±8 of the block's own code,
  checked against the full search), so a checkpoint never stores a quantised trunk: rows encode on load.
- `tf-cuda-test inkling2-mini-ref` keeps the generated positions' logits, and the float64 reference `mini_ref` can
  keep every logit (`Ref.full`): a bf16 trunk must still agree with the reference on every token; a quantised one
  passes at mean KL ≤ 0.01 nats and at most one differing pick. `trunk-ab.sh` runs that gate plus the 1- and 4-stream
  decode bench per mode on TOP's GB10.
