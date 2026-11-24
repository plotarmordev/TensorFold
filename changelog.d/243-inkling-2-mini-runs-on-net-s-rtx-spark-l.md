# Inkling 2 Mini on the RTX Spark N1X laptop (native Windows ARM64): the Windows fixes and its run steps

- Brings 189's Windows ARM64 fixes onto 188's Inkling 2 Mini engine: `__noinline__` in `glm_index_rope.cuh` and
  `/wd4819` for nvcc with cl.exe, POSIX file modes and COM `@alignCast` in the server, no `tf-muse` on Windows, and
  `compile-native-cuda-windows` checking the sm_121 aarch64 product.
- `docs/recipes/inkling-2-mini-cuda/n1x-run.ps1` runs Mini on the laptop through `codync@rtx-spark` in steps of under
  10 minutes (nvcc's kernel images, the contract images `tf-cuda-test` embeds, `tf-cuda-test` alone through the new
  `zig build tf-cuda-test` step, the server, 188's exactness checks, the reference's planted control, 1/4-stream
  serving timed by the server's own `done` lines with the WDDM pool's peak), with no Python, installs or downloads; each
  build step logs zig's output; it queries the N1X by name, as the laptop's
  `nvidia-smi` also lists an NPU. Run 6 on the laptop: all seven steps exit 0, every check passes (32 of 32 greedy
  tokens equal the float64 reference), 32.3 tok/s at one stream and 115.8 total at four, first token 0.099 s, pool
  peak 23,324 MiB.
- `tf-cuda-test inkling2-mini-ref` exits non-zero when its planted control differs, as every other failure does; the
  GB10 and N1X scripts require that exit and the plant's differing tokens, so a load or CUDA failure cannot pass it.
  The kernel contract says the MLP128 tile is served only on sm_120 instead of blaming the cluster path on sm_121.
- CUDA kernel images are keyed by nvcc's own bytes instead of a side-effect `nvcc --version` capture, which was
  rewritten every build and so missed every image's cache: each warm build ran nvcc on all of them again.
