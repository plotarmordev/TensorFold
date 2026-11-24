# Inkling 2 Mini: what Unsloth's llama.cpp adds, and what TensorFold still lacks

- `docs/inkling-2-mini-strata-study.md` reads Unsloth's shipped tree (`b12651-mix-c7059d6`: mainline b12651 plus 43
  pins) against b12651's source: its Inkling gain is seven fused CUDA kernels (conv chain, clip, SiTU, depth mix,
  relative bias in attention, elementwise chains, banded attention), about 7,300 → 3,400 kernels a token. TensorFold's
  Mini engine already fuses all but the loop controller's gate. It marks Unsloth's and Strata's levers in or missing
  with an expected GB10 gain, finds Mia's "Derisked" weights to be a refusal edit at unchanged bytes, and names the
  biggest missing lever (an MXFP8/NVFP4 trunk, +34–58%), not built here because Mini's engine is not on `main`. Inco's
  Splash 1.7.2 joins the list (Unsloth's UD-Q4_K_M weights and a DFlash2 block-diffusion drafter); cards 252–255
  and 258 carry the missing levers.
- `docs/recipes/inkling-2-mini-llamacpp/` built mainline b12651 and Unsloth's tree on TOP and measured both at 1
  and 4 streams on card 232's UD-Q4_K_M GGUF (job `top-d561fdb4ceb9`): Unsloth 44.7/45.3 tok/s solo and 139.8/117.1
  at 4 streams over mainline's 38.4/39.2 and 85.5/88.8 — +16% solo (down from its 1.54× on 1.8 TB/s cards, under
  GB10's ≈21 ms weight-read floor) and 1.32–1.63× at 4 streams. A same-head rerun gave 121.9 on 4-stream code, so
  its lead over TensorFold's 118.7 there is 3–18%, and no pin-off run ties it to particular pins; the recipe's
  result formatter now runs on Python before 3.12.
