# MiniMax-H3-Image on the Spark: cuBLASLt's picks kept on disk

- `tf-h3-image-cuda` stores the algorithms cuBLASLt's heuristic chooses — each bf16 projection's canonical (n, k) pick and each int8 product's (rows, n, k) one — in `~/.cache/tensorfold/h3-cublaslt-algos.bin`, keyed by GPU name, CUDA driver version and cuBLASLt version. A start whose file matches skips every `cublasLtMatmulAlgoGetHeuristic` call (each pick is proven on its plan by `cublasLtMatmulAlgoCheck` first), so a cold picture pays no search.
- A missing file, a foreign key (another GPU, driver or library) and a torn or truncated file are all misses, never failures: the searches run as before and the table is rewritten whole on close (a `.part` sibling renamed into place). `TF_H3_ALGO_CACHE=off` or a path of its own steers the file.
- Host tests cover the round trip and the three miss kinds in `zig/src/families/h3/cuda_algo_cache.zig` (`zig build test-h3-algo-cache`, in `zig build test`).
