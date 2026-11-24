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
- `docs/recipes/inkling-2-mini-cuda/round64.sh` (job `top-778e43c5f797`): the stored-experts contract passes all 11
  kernels, including the 64-row `tf_nvfp4_fact_softcap` and `tf_mxfp8_staged_down_f32`, and turns all 9 controls red.
  Its served trace, `tf-cuda-test inkling2-round64`, runs 64 streams of one prompt through one shared round. In that
  job (which read only the last MoE visit) the visit held 8 items of 64 rows over 8 experts, one weight read per
  expert, and the same picks routed at tile 16 took 32. 16 versus 40 is `maxItems` over those 8 experts, a bound, not
  an observed count. Each visit now keeps its own plan counts, and the trace checks every visit's observed items
  against its experts with rows; that per-visit form has not run on TOP yet.
- Both recipe scripts log `build: N s`, then cap the measurement at `timeout 580` and log `measure: N s`.
- 64-stream decode, total tok/s (median of 3, 128 tokens): two candidate runs (`top-2884ef555d3f` at `7fa99265`,
  `top-227cd6b42fe4` at `5dda89a6`, the same kernels) against one control (`top-a7166be3615d`, 211's head plus this
  script only). Same prompt `code`: 559.0 and 560.6 vs 547.3 (+2.1%, +2.4%). Same prompt `chat`: 660.9 and 664.3 vs
  644.7 (+2.5%, +3.0%). 16 prompts x 4: 411.7 and 413.8 vs 415.2 (-0.8%, -0.3%). Every cell had 0 failed requests;
  the highest host-use sample was 57.5 GiB (sampled before and after load and after the benches, not a peak meter).
  All gains are within the 3% noise.
- Same-machine Strata comparison: unmeasured. No Strata SQ-20 figure exists from TOP; card 247's HTTP run of Strata on
  this GB10 is pending. The only SQ-20 receipt is from BOTTOM (`bottom-c5e4253968cb`, Strata's own `--bench`: 64
  synthetic prompts of 64 ids, 32 tokens, no HTTP, 332.1 tok/s at 64). It is not comparable to the TOP HTTP cells above.
