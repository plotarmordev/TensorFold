# MiniMax-H3-Image on the Mac: leaner decoder finish pass, GPU-side step tail

- The step's tail is one fused dispatch a head plus a GPU Euler: `h3_head_norm` (h3_rows.metal)
  does the final layer's modulated RMS norm and the step's output head in one launch — the normed
  row lives in threadgroup memory instead of a `normed` buffer, keeping `h3_final_norm`'s SIMD sum
  order for the inverse and `h3_rows_out`'s lane-strided dot bit for bit — and `h3_euler` applies
  `x += delta * v` on the device in `picture.euler`'s two roundings — the product goes through a
  volatile so fma contraction can't fold the mul into the add and skip the first rounding.
  `h3_final_norm`, `h3_rows_out` and the `normed` staging buffer are gone.
- `Model.forward` grew a `delta` argument and a resident mode: pass the step's pdd delta and the
  video rows stay in `video_latents` across steps (an empty `video` reads them back in), so a six-step
  picture uploads its noise once and downloads its latent once instead of round-tripping both every
  step — the ~0.1 s a step of host copies card 124's profile left on the table. `video_velocity`'s
  readback is skippable, and `model.latent` reads the stepped rows after the last step.
- The VAE decoder's pass batch is 128 tiles (was 64): a 2048² canvas's 121 tiles decode in one
  pass, the way mmh3 runs its VAE — card 130's batch on the Metal path. Its `h3d_gemm` epilogue
  already folded the bias and LayerScale rounding card 130 fused on the CUDA side.
- The kernel contract covers `h3_head_norm` (norm then head in f32) and `h3_euler` against their
  float64 formulas; the resident path is exercised by `tf-h3-image`'s free run, which now takes it.
