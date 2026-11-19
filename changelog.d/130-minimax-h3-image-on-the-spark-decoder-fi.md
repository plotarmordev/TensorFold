# MiniMax-H3-Image on the Spark: decoder finish pass fused, rows already on the GPU

- `tf-h3-image-decoder-cuda --tc` fuses the finish pass where cuBLASLt's algorithm takes the bias
  epilogue: the plain projections (qkv, w1, proj_out) get `bf16(acc + b)` in one rounding —
  `h3d_finish` mode 0 exactly — and the LayerScale projections (out, w2) write that same
  `bf16(acc + b)` into a `u` staging row that the new `h3d_res_norm` folds into the next block
  norm: `x = bf16(x + bf16(u * s))` then `xn = RMSNorm(x) * g`, the mode-1 rounding points held.
  A shape whose algorithm won't fuse the bias keeps the fp32 `acc` and `h3d_finish`, and
  `--exact`'s ordered kernels are unchanged.
- Passes batch up to 128 tiles (was 64): a 2048² canvas's 121 tiles decode in one pass, the way
  mmh3 runs its VAE. The contract suite covers `h3d_res_norm` against the float64 reference
  (mode-1 residual then the norm's reduction chain), and `selftest` still requires decode in
  passes of any size to assemble bit-identically.
- The step's final norm, output head and Euler update already ran on the GPU since card 122's
  `bd734252` (`h3_rows_in`/`h3_rows_out`/`h3_euler`): this card's share of that lever was the
  decoder-side `res_norm` fusion above; the rows' 0.15 s of kernel time is unchanged.

Measured on TOP's GB10 (`top-c328ffe998ad`, same-job A/B over `bench-130-base.patch`): a warm 2048²
picture 15.2 s against the base phase's 15.3 s, decode GPU time 0.98 → 0.83 s (finish 0.26 → 0.08 s;
the projections cuBLASLt fused keep their epilogue, the rest still run `h3d_finish`), every latent
and PNG byte-identical between phases, selftests, the same-latent bound and the Mac-picture gates
all passing.
