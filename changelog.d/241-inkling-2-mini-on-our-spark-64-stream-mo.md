# Inkling 2 Mini groups up to 64 decode rows per expert on CUDA

- Mini's multi-stream MoE plan uses 64-pair items above 16 rows, while smaller rounds keep the existing 16-pair decode.
  NVFP4 and MXFP8 stage each expert's weight block once per 64-pair item, retaining each row's ordered K sums and
  soft-capped gate-up / fp32 down outputs.
- `docs/recipes/inkling-2-mini-cuda/grouped64.sh` checks the released checkpoint, greedy float64 reference and its
  planted control, then measures 64-stream same-prompt greedy decode and 64 streams over the 16 prompts of
  `tools/bench_prompts.json` (each on 4 streams) at 128 tokens, three reps, with per-request solo token equality and
  zero failed requests required.
- On TOP's checkpoint, layers 0 (NVFP4) and 20 (MXFP8), 64 tokens sharing one top-8 pick hold 8 device items instead of
  32: one weight read per expert per round, not four. The launch bound falls from 160 to 136, and the gate-up/down
  outputs match the 16-row plan byte for byte. Greedy decode still has 0 of 32 tokens off float64, the planted control
  is still red (6 of 32), and every 64-stream reply matches its solo run (192 of 192 per cell).
- 64-stream decode, total tok/s (median of 3, 128 tokens): two candidate runs (`top-2884ef555d3f` at `7fa99265`,
  `top-227cd6b42fe4` at `5dda89a6`, the same kernels) against one control (`top-a7166be3615d`, 211's head plus this
  script only). Same prompt `code`: 559.0 and 560.6 vs 547.3 (+2.1%, +2.4%). Same prompt `chat`: 660.9 and 664.3 vs
  644.7 (+2.5%, +3.0%). 16 prompts x 4: 411.7 and 413.8 vs 415.2 (-0.8%, -0.3%). Every cell had 0 failed requests;
  the highest host-use sample was 57.5 GiB (sampled before and after load and after the benches, not a peak meter).
  All gains are within the 3% noise.
- The Desk's GB10 Strata SQ-20 receipt (`bottom-c5e4253968cb`, Strata's own `--bench`: 64 distinct synthetic prompts
  of 64 ids, 32 tokens, no HTTP) gives 332.1 tok/s at 64. That receipt's HTTP cell measured nothing. These figures
  come from different benches and hosts, so this is not a matched comparison.
