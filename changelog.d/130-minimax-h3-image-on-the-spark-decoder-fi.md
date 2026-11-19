# MiniMax-H3-Image on the Spark: decoder finish pass fused, rows already on the GPU

- `tf-h3-image-decoder-cuda --tc` drops every `h3d_finish` launch on the fast path: the plain
  projections (qkv, w1, proj_out) take cuBLASLt's bias epilogue — `bf16(acc + b)` in one rounding,
  `h3d_finish` mode 0 exactly — and the LayerScale projections (out, w2) write that same
  `bf16(acc + b)` into a `u` staging row that the new `h3d_res_norm` folds into the next block
  norm: `x = bf16(x + bf16(u * s))` then `xn = RMSNorm(x) * g`, the mode-1 rounding points held.
  A shape whose algorithm won't fuse the bias keeps the fp32 `acc` and `h3d_finish`, and
  `--exact`'s ordered kernels are unchanged.
- Passes batch up to 128 tiles (was 64): a 2048² canvas's 121 tiles decode in one pass, the way
  mmh3 runs its VAE, so a picture's launches per layer fall to mmh3's own count (~270 for the
  decode). The contract suite covers `h3d_res_norm` against the float64 reference (mode-1
  residual then the norm's reduction chain), and `selftest` still requires decode in passes of
  any size to assemble bit-identically.
- The step's final norm, output head and Euler update already ran on the GPU since card 122's
  `bd734252` (`h3_rows_in`/`h3_rows_out`/`h3_euler`): this card's share of that lever was the
  decoder-side `res_norm` fusion above; the rows' 0.15 s of kernel time is unchanged.
