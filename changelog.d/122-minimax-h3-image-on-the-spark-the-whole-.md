# MiniMax-H3-Image on the Spark: the whole step in one CUDA graph

- `tf-h3-image-cuda` captures each denoising step — rows in, the blocks' norms, projections,
  heads, attention and gate mixes, the step head and the Euler update — as one CUDA graph on a
  dedicated non-blocking stream and replays it; the AdaLN lines of every step are staged once
  and each graph points `line` at its own slice. `H3_PROFILE` keeps the eager step-by-step path
  (its per-stage waits can't be captured), and a refused capture falls back to the same launches.
- The decoder's tensor-core path (`--tc`) is now the default; `--exact` opts back into the
  fp32-order kernels and keeps their 90% block-identity selftest. `selftest` gates the default.
- `tf-h3-image-decoder-cuda` also accepts a `.rows.f32` input: it reads the video rows
  progressively (unpatchify, denormalise and upload each band, then launch a tile pass once the
  latent rows its tiles read have landed), so a rows stream can overlap the transformer's tail.
  The bench measured that overlap slower at 2048² — the banded read serialises the last step and
  the progressive host work costs more than it saves — so it still feeds the decoder the finished
  `.latent.f32`. The `.rows.f32` mode stays as an opt-in input; `.latent.f32` input and output are
  unchanged. Same-day A/B on the Spark: warm 2048² mean 15.2 s against the baseline's 15.5 s —
  the whole-step graphs take ~0.2 s off the transformer.
