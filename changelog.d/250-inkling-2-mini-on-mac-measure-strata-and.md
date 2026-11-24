# Inkling 2 Mini on the M5 Ultra: TensorFold, llama.cpp and Strata at 1, 4 and 64 streams
- `docs/recipes/inkling-2-mini-metal/engines.sh` measures each engine on its own, waiting for a 1-minute load under 10
  before each cell. End to end, TensorFold was fastest at 4 streams and on diverse prompts at 1 stream. Strata v0.2.19
  SQ-20 was fastest at 64 streams (763.6 vs 486.2 tok/s with the same prompt) and at 1 stream with the same prompt.
  llama.cpp b12655 IQ4_NL trailed in every cell. Strata was measured in a later window than the others and varied 2×
  between windows.
