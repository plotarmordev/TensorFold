# Windows ARM64: `zig build native install -Dsm=121` compiles again

- `tf-muse` is no longer built for Windows: its rank 1 follows over the POSIX TP session, which
  broke the RTX Spark N1X build with 11 errors in `std.c` and `zig/src/fabric/tp`.
- The server's speech turn (`gpu_turn.zig`) skips POSIX file modes on Windows, and DXGI's COM
  calls cast vtable entries with `@alignCast` for aarch64; both stopped `tensorfold-native`.
- `compile-native-cuda-windows` now also type-checks the sm_121 `aarch64-windows-gnu` product.
- `docs/WINDOWS.md` lists what the native N1X build still lacks.
