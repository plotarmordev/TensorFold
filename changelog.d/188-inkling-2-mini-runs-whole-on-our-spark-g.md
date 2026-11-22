# Inkling 2 Mini on CUDA: upstream #4387 with jschmied's #4364, #4365 and #4441, and a whole-model run on TOP

- Merges Ash's #4387 and jschmied's C1 (#4364), C3 (#4365) and H2 + C4 (#4441) with their commits; #4387's engine
  serves `inkling2_loop`, with jschmied's three fixes (layer-sized plan, one sliding-norm upload, raw gate and sink).
- `docs/recipes/inkling-2-mini-cuda/run-script.sh` runs the original checkpoint on TOP's GB10: #4387's serving
  checks, the float64 reference and its planted control, then first-token time, prompt speed, decode speed and
  memory at 1 and 4 streams. The recipe's README notes what RTX Spark (Windows and WSL2) needs.
- Mini prompt snapshots save and restore each key head's ring rows: the CUDA caches are head-major, and the
  row-major copy left restored prompts' logits off from a fresh prefill.
- Mini loop messages now store `[core][role][row][c]` planes, matching each controller's role offset across multi-row prompt chunks; the CUDA reference checks the same served layout.
