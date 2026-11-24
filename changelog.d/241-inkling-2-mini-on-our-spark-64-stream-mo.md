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
- `docs/recipes/inkling-2-mini-cuda/round64.sh` at `b223d41d` (jobs `top-20f0166ebc50` and `top-a9efe22b9d83`, both exit
  0): the stored-experts contract passes all 11 kernels, including the 64-row `tf_nvfp4_fact_softcap` and
  `tf_mxfp8_staged_down_f32`, and turns all 9 controls red; the checkpoint's 64-row outputs match the 16-row plan byte
  for byte. Its served trace, `tf-cuda-test inkling2-round64`, runs 64 streams of one prompt through one shared round
  and reads every MoE visit's device counts: each of the 48 visits held 8 items, one per expert with rows (one weight
  read per expert), and the last visit's picks routed at tile 16 took 32. 16 versus 40 is `maxItems` over those 8
  experts, a bound, not an observed count.
- Both recipe scripts log `build: N s`, then cap the measurement at `timeout 580` and log `measure: N s`.
- 64-stream decode, total tok/s (median of 3, 128 tokens), from `grouped64.sh`: two candidate runs (`top-2884ef555d3f`
  at `7fa99265`, `top-227cd6b42fe4` at `5dda89a6`, the same kernels) against one control (`top-a7166be3615d`, 211's
  head plus this script only). Same prompt `code`: 559.0 and 560.6 vs 547.3 (+2.1%, +2.4%). Same prompt `chat`: 660.9
  and 664.3 vs 644.7 (+2.5%, +3.0%). 16 prompts x 4: 411.7 and 413.8 vs 415.2 (-0.8%, -0.3%). Every cell had 0 failed
  requests; the highest host-use sample was 57.5 GiB (sampled before and after load and after the benches, not a peak
  meter). All gains are within the 3% noise. The float64, planted-control and solo-equality gates above last ran in
  those two candidate jobs, at `7fa99265` and `5dda89a6`; `b223d41d` changes the per-visit plan counts, the trace and
  the recipes after them, and has not rerun `grouped64.sh`.
- Same-machine Strata SQ-20 comparison: unmeasured here. It is owned by card "ink moe64 port", which ports this grouped
  tile onto main's engine. No claim against Strata is made from the cells above.
