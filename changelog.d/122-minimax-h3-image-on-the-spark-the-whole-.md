# MiniMax-H3-Image on the Spark: the whole step in one CUDA graph

- `tf-h3-image-cuda` captures each denoising step — rows in, the blocks' norms, projections,
  heads, attention and gate mixes, the step head and the Euler update — as one CUDA graph on a
  dedicated non-blocking stream and replays it; the AdaLN lines of every step are staged once
  and each graph points `line` at its own slice. `H3_PROFILE` keeps the eager step-by-step path
  (its per-stage waits can't be captured), and a refused capture falls back to the same launches.
- The decoder's tensor-core path (`--tc`) is now the default; `--exact` opts back into the
  fp32-order kernels and keeps their 90% block-identity selftest. `selftest` gates the default.
- The decoder-side rows overlap was built and measured, then cut: `top-27eefb4ad8de` ran the last step
  streaming video rows out band by band while the decoder read them progressively, and warm 2048² came
  out 15.7 s — the banded read serialised the step's tail and the progressive host decode cost 3.6 s vs
  1.3 s for the bulk path — so the decoder keeps its `.latent.f32` input and output unchanged. The
  whole-step graphs + tc default measured warm 2048² mean 15.2 s against the same-day baseline's 15.5 s
  (~0.2 s off the transformer from graph replay) — kept.
