# MiniMax-H3-Image on the Mac: int8 weights packed in fragment order

- Bind quantizes every block's four int8 projections straight into the order the tensor units' right operand takes
  them, and `h3_gemm_32x64_i8` loads each lane's B fragments as one 16-byte read instead of eight scalar reads.
  Outputs are byte for byte main's: the warm 2048 pilot latent stays `776b443a…`.
- Replayed alone, the four products run 11-52% faster; served in card 197's quiet pairs (each side started under
  load 8), a warm 2048 picture's six steps take 7.63 s against main's 9.34 s, and bind takes 0.19 s longer.
