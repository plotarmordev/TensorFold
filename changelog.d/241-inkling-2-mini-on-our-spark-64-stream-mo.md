# Inkling 2 Mini groups up to 64 decode rows per expert on CUDA

- Mini's multi-stream MoE plan uses 64-pair items above 16 rows, while smaller rounds keep the existing 16-pair decode.
  NVFP4 and MXFP8 stage each expert's weight block once per 64-pair item, retaining each row's ordered K sums and
  soft-capped gate-up / fp32 down outputs.
- `docs/recipes/inkling-2-mini-cuda/grouped64.sh` checks the released checkpoint, greedy float64 reference and its
  planted control, then measures 64-stream same-prompt and distinct-prompt greedy decode at 128 tokens, three reps,
  with per-request solo token equality and zero failed requests required.
