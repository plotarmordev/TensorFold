# Inkling 2 Mini CUDA: the DSpark draft head
- Load the DSpark head from `mtp.safetensors` and run it on CUDA (`inkling2_dspark`): committed taps fused into per-layer context rings, the three-layer block on the target's bf16 lane GEMM, then seven Markov steps whose greedy picks feed the next step on the device, with no host round trip between drafts.
- Check every draft step's logits against a float64 host reference of the same head on seeded context rows (`tf-cuda-test inkling2-dspark`, `docs/recipes/inkling-2-mini-cuda/dspark.sh`), with a planted context-blind reference that must go red.
