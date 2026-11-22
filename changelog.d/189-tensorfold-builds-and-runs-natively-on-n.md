# Windows ARM64: `zig build native install -Dsm=121` compiles again

- `tf-muse` is no longer built for Windows: its rank 1 follows over the POSIX TP session, which
  broke the RTX Spark N1X build with 11 errors in `std.c` and `zig/src/fabric/tp`.
- The server's speech turn (`gpu_turn.zig`) skips POSIX file modes on Windows, and DXGI's COM
  calls cast vtable entries with `@alignCast` for aarch64; both stopped `tensorfold-native`.
- `compile-native-cuda-windows` now also type-checks the sm_121 `aarch64-windows-gnu` product.
- `glm_index_rope.cuh` uses CUDA's `__noinline__`, not GCC's `__attribute__((noinline))`, which nvcc
  with cl.exe rejected in `glm_glue` and `glm_index`; Windows fatbins pass `/wd4819`, the flag that
  showed that error under the code-page warnings burying it.
- `docs/WINDOWS.md` records the N1X laptop's build, `tf-cuda-test` counts, the kernel-contract run that
  ended before its SUMMARY (12 FAIL lines, all MiniMax-H3 image kernels), Qwen3.5-2B tok/s, and what
  Inkling 2 Mini still needs there.
- The kernel contract's `h3p_attention` takes gate 4's 4-row cut of its 77-row shape (its keys are its rows; it
  refused fewer rows than K on every OS), and MLP128's N/A line names its sm_120-only tile, not the sm_90 cluster.
