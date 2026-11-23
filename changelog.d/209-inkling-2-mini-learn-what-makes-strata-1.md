# Inkling 2 Mini: what makes Strata fast, and what TensorFold lacks

- `docs/inkling-2-mini-strata-study.md` reads Strata (MIT, Niko1221/Strata v0.2.19–v0.2.20) and explains its Inkling 2
  Mini numbers lever by lever: the W=4 on-GPU DSpark beam tree, the greedy ids-only readout, the sm_120a/121a fp4
  tensor-core experts, the `.strata` SQ packs, the 48-visit CUDA graphs, 64-slot batching and FP8 KV. It marks each
  lever against TensorFold, with the file a port would touch and a gain from GB10's 273 GB/s ceiling. On Windows ARM64
  it separates Strata's absent build from llama.cpp's Windows ARM64 CUDA builds (Mini from b12600, first fully correct
  in b12641) and card 243's unmerged N1X run of TensorFold's Mini (checks pass, no tok/s yet), and orders the
  follow-up cards for GB10 and the RTX Spark laptop.
- `docs/recipes/strata-inkling/` stages Strata's own linux-aarch64 engine and its SQ-10 pack on TOP and benches them
  at 1, 4 and 64 streams through `swarm gpu-run`. On TOP's GB10, Strata does 74.4 / 97.1 / 382.5 tok/s at
  1 / 4 / 64 streams; TensorFold's Mini does 32.5 at 1, 117.8 at 4, and 589.8–593.0 at 64 in a separate HTTP
  receipt with different context, prompts and generation length. Drafting's solo contribution remains unmeasured.
