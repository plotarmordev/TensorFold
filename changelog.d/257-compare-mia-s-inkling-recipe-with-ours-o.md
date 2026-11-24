# Inkling 2 Mini on CUDA: a comparison page against Mia's DGX Spark recipe, with the shared 1/2/4/8-stream harness

- `docs/recipes/inkling-2-mini-cuda/README.md` sets our merged `inkling2_loop` engine beside Mia's AI Lab's DGX
  Spark recipe (TensorFold `dc2ca8a1` + patches 0001-0017: CUDA DSpark drafts, fp8 KV and attention, Blackwell
  FP4/MXFP8 prefill tensor cores, vision, structured output) feature by feature, and our measured N1X/GB10 rows
  beside her published sparkDash figures, every cross-hardware row labelled non-comparable. Her recipe runs
  only in its Linux container and was not run.
- `run-script.sh` (TOP's GB10) and `n1x-run.ps1` (the RTX Spark laptop, native Windows ARM64) now measure
  1, 2, 4 and 8 greedy streams plus a prefill probe, TTFT, peak pool memory, concurrent-versus-solo output
  agreement and failures under one harness and one 9/10-minute budget a step.
