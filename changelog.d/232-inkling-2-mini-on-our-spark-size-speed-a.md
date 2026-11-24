# Inkling 2 Mini against llama.cpp on TOP's GB10

- `docs/recipes/inkling-2-mini-cuda/compare.sh` builds llama.cpp b12611's CUDA `llama-server` from the Desk's staged
  sources and serves Unsloth's UD-Q4_K_M GGUF at the CUDA receipt's settings, with each engine's file bytes,
  `peak-memory.py`'s server peak, and greedy and teacher-forced agreement with the float64 reference.
- The recipe README sets it beside TensorFold (job `top-b77911768550`): 39.1 against 32.9 tok/s at one stream,
  87.6 against 120.1 at four, 15.47 against 18.78 GiB of weights; both reproduce the reference's 32 greedy tokens.
