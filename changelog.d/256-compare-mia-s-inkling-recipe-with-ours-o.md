# Inkling 2 Mini: Mia's DGX Spark recipe pinned, and its seventeen patches priced against ours

- `docs/recipes/inkling-2-mini-cuda/README.md` pins MiaAI-Lab's `Inkling-2-Mini-DGX-Spark-TensorFold` at `6061fa89`
  (TensorFold `dc2ca8a1` + patches `0001`–`0017`, `nvcr.io/nvidia/pytorch:26.07-py3`, Zig 0.17.0), separates the upstream
  checkpoint (`7a67ae23`, shard `65e60f51`, MTP `7c3b6513`) from Blackfrost's Derisked derivative the recipe serves only
  with `RERISKED=1`, and records its settings (`262144`, `PARALLEL=8`, fp8 KV, DSpark drafts, opt-in FP8 attention and
  prefill tensor cores) and its harness (sparkDash's dashboard panel; the repository's `tools/bench_streams.py` imports
  a `tools/bench.py` it does not ship, so the published table is not regenerable from the recipe's files).
- Its seventeen patches are checked one by one against `c99eb630d`: 0001, 0002, 0013, 0014 and 0016 are partial, the
  rest missing. The missing exact paths are 0014's 4,096-row prompt passes and wide graph cache, 0009/0005's fused
  router and projections, 0007's two-warp expert split, 0008's certified lazy head and warp top-k, 0011's `cp.async`
  decode prefetch, 0001–0002's verify window and served DSpark head, and 0003's fp8 KV; 0012 and 0015 are the lossy
  opt-ins. No gain is claimed before a paired run prices it.
- `mia-compare.sh` and `mia-compare.py` pair the two builds on one GB10: same checkpoint with its SHA-256 gates, greedy
  decode at 1/2/4/8 streams, prefill 4k–128k with a fresh text a repetition, six agreement prompts by token and text sha,
  peak RSS and GPU memory a phase, medians of three. Both arms run `--context 131072 --parallel 8`, bf16 KV and drafts
  off, because our Mini CUDA head has no fp8 KV and refuses `--drafter`. The receipt is pending a Desk-serialized GPU
  window: two builds and a measurement phase above the slot's 600 s bound.
