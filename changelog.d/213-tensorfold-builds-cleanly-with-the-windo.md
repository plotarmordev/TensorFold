# TensorFold builds cleanly with the Windows CUDA compiler

- `glm_index_rope.cuh` marks the indexer rope's `turn` with CUDA's portable `__noinline__` instead of
  GCC's `__attribute__((noinline))`, which MSVC-hosted nvcc rejects; `tf-muse` and the CUDA runners
  now build on Windows without local patches. Linux codegen is unchanged.
