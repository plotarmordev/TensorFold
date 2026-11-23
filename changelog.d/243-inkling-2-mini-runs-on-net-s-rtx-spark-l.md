# Inkling 2 Mini on the RTX Spark N1X laptop (native Windows ARM64): the Windows fixes and its run steps

- Brings 189's Windows ARM64 fixes onto 188's Inkling 2 Mini engine: `__noinline__` in `glm_index_rope.cuh` and
  `/wd4819` for nvcc with cl.exe, POSIX file modes and COM `@alignCast` in the server, no `tf-muse` on Windows, and
  `compile-native-cuda-windows` checking the sm_121 aarch64 product.
- `docs/recipes/inkling-2-mini-cuda/n1x-run.ps1` runs Mini on the laptop through `codync@rtx-spark` in three short steps
  (build, 188's exactness checks, 1/4-stream serving with first token and the WDDM pool's peak), with no Python, installs
  or downloads; it queries the N1X by name, as the laptop's `nvidia-smi` also lists an NPU. Run 1 on the laptop: the
  build and every check pass (32 of 32 greedy tokens equal the float64 reference); the tok/s receipt is next.
- `tf-cuda-test inkling2-mini-ref` exits non-zero when its planted control differs, as every other failure does, and
  the GB10 script requires that; the kernel contract says the MLP128 tile is served only on sm_120 instead of blaming
  the cluster path on sm_121.
