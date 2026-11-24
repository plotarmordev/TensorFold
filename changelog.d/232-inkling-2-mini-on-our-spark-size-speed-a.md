# Inkling 2 Mini against llama.cpp on TOP's GB10

- `docs/recipes/inkling-2-mini-cuda/compare.sh` builds llama.cpp b12611's CUDA `llama-server` from the Desk's staged
  sources and serves Unsloth's UD-Q4_K_M GGUF at the CUDA receipt's settings, with each engine's file bytes,
  `peak-memory.py`'s server peak, and greedy and teacher-forced agreement with the float64 reference.
- The recipe README sets it beside TensorFold (jobs `top-b77911768550`, `top-c0075bd36227`): 39.1 against 32.9 tok/s
  at one stream, 87.6 against 120.1 at four, 15.47 against 18.78 GiB of weights, and 106.4 against 196.8 tok/s
  uncached at one-stream prompt processing. llama.cpp's free-running text matches the reference continuation when
  re-tokenized, but its teacher-forced picks match 31 of 32 reference tokens; TensorFold matches 32 of 32.
