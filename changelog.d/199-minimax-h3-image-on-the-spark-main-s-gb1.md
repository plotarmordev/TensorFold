# MiniMax-H3-Image on CUDA: main's GB10 kernel-contract suites green again
- `h3p_scatter_add` and `h3p_splice` contract fixtures: gate 1's raised row count scattered to row `2i`, one row
  past `x` in both the host reference and the device buffer (glibc's `corrupted size vs. prev_size`, then abort); the
  rows now spread inside `x`.
- `h3m_sinusoid`'s shapes bound carries the f32 angle's own rounding through `cos` and `sin`: `(4|x| + 5)` ulps of
  the angle, derived, not a loosened gate.
- Limits records: `h3_norm`, `h3_quantize`, `h3_norm_quant`, `h3_norm_quant_fp4` and `h3p_norm` hold 1024 static
  shared bytes, the TMA header's `extern __shared__ __align__(1024)` buffer padding every kernel in its unit;
  `h3_swiglu_rows` gets its sm_121 usage (22 registers).
- `check-199.sh` runs each H3 suite in its own process, then the H3 images' five-SM limits audit.
