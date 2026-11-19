# MiniMax-H3-Image on the Spark: the whole step in one CUDA graph

- `tf-h3-image-cuda` captures each denoising step — rows in, the blocks' norms, projections,
  heads, attention and gate mixes, the step head and the Euler update — as one CUDA graph on a
  dedicated non-blocking stream and replays it; the AdaLN lines of every step are staged once
  and each graph points `line` at its own slice. `H3_PROFILE` keeps the eager step-by-step path
  (its per-stage waits can't be captured), and a refused capture falls back to the same launches.
- The decoder's tensor-core path (`--tc`) is now the default; `--exact` opts back into the
  fp32-order kernels and keeps their 90% block-identity selftest. `selftest` gates the default.
- The decoder overlaps the transformer's tail: the last step streams the target's video rows
  out band by band on the same stream as the graph, and `tf-h3-image-decoder-cuda` reads a
  `.rows.f32` input progressively (the bench hands it a fifo), unpatchifies, denormalises and
  uploads each band and launches every tile pass as soon as the latent rows its tiles read have
  landed. `.latent.f32` input and output are unchanged.
