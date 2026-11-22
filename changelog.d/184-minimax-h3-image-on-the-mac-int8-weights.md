# MiniMax-H3-Image on the Mac: int8 weights packed in fragment order

- Bind quantizes every block's four int8 projections straight into the order the tensor units' right operand takes
  them, and `h3_gemm_32x64_i8` loads each lane's B fragments as one 16-byte read instead of eight scalar reads.
  Outputs are byte for byte main's: the warm 2048 pilot latent stays `776b443a…`.
- Replayed alone, the four products run 11-52% faster; served, a warm 2048 picture's six steps take 7.71-7.81 s
  against main's 8.95-9.33 s at load 5.6 (head's session started at load 20), and bind takes 0.2 s longer.
