# TensorFold builds cleanly with the Windows CUDA compiler

- `glm_index_rope.cuh` marks the indexer rope's `turn` with CUDA's portable `__noinline__` instead of
  GCC's `__attribute__((noinline))`, which MSVC-hosted nvcc rejects. Linux codegen is unchanged.
- The Windows CUDA install compiles: DXGI's vtable entries are cast with `@alignCast`, the speech lock sets and
  checks mode bits only on POSIX, and `tf-qwen3-asr` times with `std.Io`'s clock instead of `clock_gettime`.
- `tf-muse` builds on Windows: one rank and `MUSE_PAIR` run there, while `TF_TP_WORLD` and `follow` refuse with
  `MuseSparkTpPlatform`, as the native server already does, since fabric/tp's sockets and verbs are POSIX.
- The H3-Image "selftest scratch" test opens its folder iterable, so `zig build test` lists it on Linux
  (it failed on TOP's Spark at the merge base too).
- A clean checkout at `22c92fff` builds 230/230 steps, including `tf-cuda-test.exe`, `tensorfold-native.exe` and
  `tf-muse.exe`, with MSVC-hosted nvcc 13.5 for `sm_121`. Receipt: `docs/receipts/windows-cuda-build-22c92fff/`.
