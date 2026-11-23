# MiniMax-H3-Image on the Spark: per-128-of-K int8 `out`/`fc2` on the TMA product measured and reverted

`out` and `fc2` ran on card 146's TMA int8 product with a scale per 128 of K on both operands. `h3_quantize_blocks`
scaled each 128-wide chunk, and `h3_gemm_int8_tma_blocks` folded each K block's exact int32 sums into fp32 in
registers. The same-job A/B (job `top-639918032e5e`, at `095874ff`) measured warm 2048² at 13.37 s against main's
14.14 s, with the transformer at 10.04 s against 10.81 s. But the pictures failed the gates: unseen bakery
(23.42 dB, mean diff 7.77) and orchard (23.17 dB, 8.17) against `--bf16`, and the fox against episode 1's Mac
picture (0.9910 cosine, 26.26 dB, 5.17, where main holds 28.69 dB and 4.01). So the code is reverted and `out` and
`fc2` stay bf16. The code is at `8027eb54`; receipts are in `docs/recipes/minimax-h3-cuda.md`
under card 176.
