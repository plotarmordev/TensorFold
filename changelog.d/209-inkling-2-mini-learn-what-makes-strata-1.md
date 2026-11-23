# Inkling 2 Mini: what makes Strata fast, and what TensorFold lacks

- `docs/inkling-2-mini-strata-study.md` reads Strata (MIT, Niko1221/Strata v0.2.19–v0.2.20) and explains its Inkling 2
  Mini numbers lever by lever: the W=4 on-GPU DSpark beam tree, the greedy ids-only readout, the sm_120a/121a fp4
  tensor-core experts, the `.strata` SQ packs and the 48-visit CUDA graphs. It marks each lever against TensorFold,
  with the file a port would touch and a gain from GB10's 273 GB/s ceiling, states that nobody ships a Windows ARM64
  Inkling engine, and orders the follow-up cards for GB10 and the RTX Spark laptop.
- `docs/recipes/strata-inkling/` stages Strata's own linux-aarch64 engine and its SQ-10 pack on TOP and benches them
  at 1, 4 and 64 streams through `swarm gpu-run`. The staging has not run yet, so no Strata GB10 row exists.
