# MiniMax-H3-Image on the Spark: decoder attention with its tiles in registers

- `h3d_flash` computes `q·kᵀ` and `p·v` with `mma.sync` m16n8k16 fragments held in registers: the
  scores' accumulators become the weights' A fragments directly, and only the keys and values pass
  through shared memory (swizzled for `ldmatrix`). Shared memory falls from 40 KiB to 16 KiB a block,
  with 128–130 registers.
- Warm 2048² decoder attention 0.19 s → 0.03 s, decoder GPU 0.78 s → 0.62 s (job `top-6b195e022759`),
  latents byte-identical to main, same-latent reference 56.43 dB (relative 3.26e-3), contract gates 1-4 pass.
