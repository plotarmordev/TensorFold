# Windows ARM64: `zig build native install -Dsm=121` compiles again

- `tf-muse` is no longer built for Windows: its rank 1 follows over the POSIX TP session, which
  broke the RTX Spark N1X build with 11 errors in `std.c` and `zig/src/fabric/tp`.
- The server's speech turn (`gpu_turn.zig`) skips POSIX file modes on Windows, and DXGI's COM
  calls cast vtable entries with `@alignCast` for aarch64; both stopped `tensorfold-native`.
- `compile-native-cuda-windows` now also type-checks the sm_121 `aarch64-windows-gnu` product.
- `glm_index_rope.cuh` uses CUDA's `__noinline__`, not GCC's `__attribute__((noinline))`, which nvcc
  with cl.exe rejected in `glm_glue` and `glm_index`; Windows fatbins pass `/wd4819`, the flag that
  showed that error under the code-page warnings burying it.
- `docs/WINDOWS.md` records the N1X laptop's clean native build and eight passing `tf-cuda-test` suites. A prior kernel-contract run aborted at gate 5 on a missing LP64-mangled symbol; the contract now counts each missing image symbol as one limit failure and continues to its SUMMARY. Main's derived `h3m_sinusoid` bound is included by merge, but the updated contract and nvcc 13.5 resource records still need a fresh Windows receipt. Model serving and tok/s are outside this card.
- The kernel contract's `h3p_attention` takes gate 4's 4-row cut of its 77-row shape (its keys are its rows; it
  refused fewer rows than K on every OS), and MLP128's N/A line names its sm_120-only tile, not the sm_90 cluster.
- `h3p_scatter_add` and `h3p_splice`'s contract rows stay inside their buffer: gate 1's 5-row shape wrote past it
  and ended `kernel-contract` before its SUMMARY ("double free or corruption" on Linux, exit code 5 on Windows).
- The `h3d` and `h3p` contract references end unrounded, so a tie the kernel's fp32 rounds the other way is half an
  output step off, as the bound allows, not a whole one; `h3p_norm` qwen's is unrounded throughout and `h3p_heads`
  rotary declares its four bf16 points. `h3p_norm` records the 1024 static shared bytes `h3_int8_tma.cuh`'s
  aligned shared array gives it.
- `tf-qwen3-asr` times its stages with `std.Io`'s awake clock, not POSIX `clock_gettime`, the one step of 228 that
  failed the N1X build at `2d0a39ce3`.
- The kernel contract's `d3_cfg_mix` takes gate 1's five rows as more logits (it refused any row count but one, on
  every OS). `kernel-contract` prints its SUMMARY when a suite or gate 5 stops it too, marked "(stopped early)", and
  a COUNT line after each kernel; `zig build kernel-contract -Dcontract-only=WORD` runs the kernels so named.
- `tf-cuda-test kernel-contract I/N [WORD...]` runs every N-th kernel and gate 5 launch, so N processes side by side
  put a contract run on N cores (it was one thread: 2 h 40 min on the N1X); `contract-job.sh` runs its four on TOP.
- The MOSS-Image conv contract includes fp32 `expf` overflow at a stress sum of -89 in its signed silu
  interval; the Idol decoder attention reference rounds P against each tile's running max before rescaling,
  matching the kernel's 64-key tiles at the 65-key edge. Both corrections await a Windows receipt.
