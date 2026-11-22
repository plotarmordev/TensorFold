# MiniMax-H3-Image on the Spark: dense attention's QK and PV on the INT8 tensor cores, measured and reverted

- `h3_attention_i8` ran the dense image attention with both products on `mma.sync.aligned.m16n8k32`: QK^T on
  s8×s8, P·V on u8×s8, the fp32 online softmax and its int32 sums exact. Q and K rows were quantized at their
  absmax/127, V per tile and channel at its live keys' absmax/127, and the probabilities as `round(p * 255)` codes
  staged through shared memory; `h3_heads_quant` wrote them from `h3_heads`' bf16 tiles. It took 168 registers a
  thread on all five recorded SMs against `h3_attention`'s 217, and the kernel contract's gates 1-4 passed on the
  pair (job `top-afbef35083c5`).
- Same-job A/B on GB10 (job `top-28126dd9a5e7` at `fa094cb8e`, the base patch reverse-applied to main's `zig/`
  tree): the attention stage took 2.26–2.28 s a picture against main's 1.76–1.81 s, and every picture missed the
  gates — latent cosine 0.9242–0.9944, PSNR 16.57–26.04 dB, mean diff 7.60–20.57 — on the fox and three unseen
  prompts, with every selftest passing on both sides.
- Reverted in full: the dense path stays `h3_attention`. The u8 codes round every probability below 1/510 to zero,
  which loses the diffuse rows' mass (`fb5090ae`'s e4m3 probabilities measured 0.9882 on the same model), and int8
  fragments are 4-byte shared reads where `ldmatrix4` moves the bf16 tiles, so the narrower mma returns less than
  the quantization pass and the loads cost. The registers' win and the measurement are in
  `docs/recipes/minimax-h3-cuda.md` (card 179).
