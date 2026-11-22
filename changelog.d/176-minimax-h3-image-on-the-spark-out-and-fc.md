# MiniMax-H3-Image on CUDA: `out` and `fc2` int8 on the TMA product at a scale per 128 of K
- The `--int8` default now runs `out` and `fc2` on card 146's TMA product too. A new `h3_quantize_blocks`
  gives every 128-wide chunk of K its own scale, and `h3_gemm_int8_tma_blocks` folds each block's exact int32
  sums at those scales in registers. `qkvc` and `fc1` keep their per-row path.
- On TOP (job `top-639918032e5e`), a warm 2048² picture took 13.37 s against main's 14.14 s, with the
  transformer at 10.04 s against 10.81 s. Against the same job's `--bf16` picture, the unseen glacier prompt
  passes the gates, while bakery and orchard fail on both sides. The fox misses episode 1's gates (26.26 dB,
  mean 5.17), which main holds.
