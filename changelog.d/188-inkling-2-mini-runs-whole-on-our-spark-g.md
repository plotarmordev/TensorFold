# Inkling 2 Mini on CUDA: jschmied's #4441 engine registered, with #4387's checks and a whole-model run on TOP

- Merges Ash's #4387 and jschmied's C1 (#4364), C3 (#4365) and H2 + C4 (#4441) with their commits. `inkling2_loop`
  is now served by #4441's engine (graphs, fused input projections, every stream's rows in one forward); #4387's
  separate loop engine is removed, so CUDA no longer advertises structured output it refused.
- #4387's serving checks (device draws against host selection, fresh against resumed prompts, two streams in shared
  rounds against each alone) run against #4441's engine. Copy drafting is not checked: that engine has no draft path yet.
- `docs/recipes/inkling-2-mini-cuda/run-script.sh` runs the original checkpoint on TOP's GB10: the serving checks,
  the float64 reference and its planted control, then first-token time, prompt speed, decode speed, memory
  (`/proc/meminfo`) and the bandwidth limit at 1 and 4 streams; 4 streams must give 1.5 times one stream's total.
  The recipe's README notes what RTX Spark (native Windows and WSL2) needs.
- Metal Mini admission budgets a target state for every stream, so concurrent full-window requests no longer run out
  of state; the Metal checkpoint check reads the first logprob row after the first token.
