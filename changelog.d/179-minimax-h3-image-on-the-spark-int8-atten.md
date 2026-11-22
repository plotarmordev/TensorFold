# MiniMax-H3-Image on the Spark: dense attention's QK and PV on the INT8 tensor cores

- `h3_attention_i8` (`h3_attention_int8.cuh`) runs the image transformer's dense attention with QK^T on
  `mma.sync.aligned.m16n8k32.s8.s8.s32` and P·V on `m16n8k32.u8.s8.s32`, fp32 online softmax and exact int32
  sums: Q and K rows at their absmax/127, V per tile and channel at its live keys' absmax/127, probabilities as
  `round(p * 255)` codes in shared memory at 80-byte rows. `h3_heads_quant` writes the codes and scales from
  `h3_heads`' bf16 tiles once a parity a block, so the attention stage carries it. The routed path keeps
  `h3_attention`.
- 168 registers a thread on all five recorded SMs against `h3_attention`'s 217 on sm_120/121 (`h3_heads_quant`
  80, 86 on sm_120/121), and the kernel contract's gates 1-4 pass on the pair (job `top-afbef35083c5`).
