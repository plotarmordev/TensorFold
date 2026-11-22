# MiniMax-H3-Image on the Spark: per-128-of-K int8 `out`/`fc2` on the TMA product measured and reverted

`out` and `fc2` ran on card 146's TMA int8 product with a scale per 128 of K on both operands. `h3_quantize_blocks`
scaled each 128-wide chunk, and `h3_gemm_int8_tma_blocks` folded each K block's exact int32 sums into fp32 in
registers. The same-job A/B (job `top-639918032e5e`, at `095874ff`) measured warm 2048² at 13.37 s against main's
14.14 s, with the transformer at 10.04 s against 10.81 s. But the fox missed episode 1's Mac gates: 0.9910 cosine,
26.26 dB, mean diff 5.17, against main's 28.69 dB and 4.01. An opt-in below the gates has no use, so the code is
reverted and `out` and `fc2` stay bf16. The code is at `8027eb54`; receipts are in `docs/recipes/minimax-h3-cuda.md`
under card 176.
