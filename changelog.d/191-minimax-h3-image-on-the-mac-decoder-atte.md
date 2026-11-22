# MiniMax-H3-Image on the Mac: decoder attention with its tiles in registers

- `h3d_attention` computes `q·kᵀ` and `p·v` on 8 x 8 simdgroup matrices: each simdgroup's scores become its weights in
  registers and feed p·v directly, and only the keys and values pass through threadgroup memory (8,704 bytes, 32 keys
  at a time). 32 queries a threadgroup of 128; one kernel for every Apple GPU.
- Warm 2048x2048 on an M5 Ultra, same session against main: decoder attention 0.16-0.20 s → 0.05-0.06 s, decoder GPU
  0.56-0.61 s → 0.45-0.49 s; the same latent decodes to 56.54 dB of main's PNG (card 80's floor is 28.2 dB).
