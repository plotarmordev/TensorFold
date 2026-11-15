# MiniMax-H3-Image benchmark run on the Spark

- `docs/recipes/minimax-h3-cuda/bench-script.sh` runs through the Desk's GPU door like `run-script.sh`, with the same prompt, seed 42 and steps at 2048² and 4096². It records each CUDA tool's selftest PASS or FAIL and renders anyway. Each size gets one cold picture and three warm repeats, with per-stage times. `nvidia-smi` samples give peak memory, average watts and energy per picture. `run-script.sh` keeps its selftest gate.
- `tools/zig/h3_image_bench.py` writes `SUMMARY.md`, headed by the selftests' verdict (a failing build's timings are marked as not the Spark's result), with every number, the commit and the settings, including after a failed stage. `test_h3_image_bench.py` covers its parsing without a GPU.
