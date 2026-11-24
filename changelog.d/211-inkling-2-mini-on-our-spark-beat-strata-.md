# Inkling 2 Mini on CUDA: a hill-climb script and its first GB10 receipt

- `docs/recipes/inkling-2-mini-cuda/hillclimb.sh` now measures same-prompt c1/c64 and 64 distinct prompts from a
  committed, hashed set. Every request must produce a valid timing; failed arms are logged and the server stops on exit.
- On TOP's GB10 (job `top-975292aaf4c0`) the stored checkpoint's no-draft engine gave 32.6–32.9 tok/s at one
  stream and 590–597 total at 64 on same-prompt runs, with no request failed and 0/32 tokens off float64. Both
  tested levers remained inside noise and were rejected; no mixed-prompt result or kept speed lever has been
  established.
- Strata's published SQ-10 and SQ-20 numbers belong to different packs and are not matched to this checkpoint's
  HTTP workload. The 49.2 tok/s bandwidth ceiling applies only to our larger checkpoint. The one-stream bar and
  the published SQ-20 64-stream bar remain unmet; cards 215/219 are prospective work, not evidence for this card.
  Strata on this GB10 is pending: card 247's first HTTP run returned mostly empty replies.
- Removed three obsolete CUDA fatbins (`inkling2_experts`, `inkling2_modelopt`, `inkling2_vision`) and their unused
  host modules and test-only contract records; the serving engine continues to use shared CUDA experts.
