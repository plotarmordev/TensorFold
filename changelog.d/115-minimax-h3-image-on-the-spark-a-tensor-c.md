# MiniMax-H3-Image: a tensor-core decoder path on the Spark

- `tf-h3-image-decoder-cuda`'s `--tc` opts the VAE decoder into tensor cores: every projection runs
  through cuBLASLt with fp32 sums (the same `h3d_finish` keeps the bias and LayerScale rounding
  points), and attention runs as batched q·kᵀ and p·v products around the new `h3d_softmax` and
  `h3d_unheads` kernels. The default fp32-order kernels and the 90% block-identity selftest are
  unchanged; `selftest --tc` gates the fast path against the same float64 reference, and the
  picture gates are the same-latent PSNR and the PSNR/mean-diff checks against the Mac picture.
