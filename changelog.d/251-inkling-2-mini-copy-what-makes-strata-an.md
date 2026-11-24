# Inkling 2 Mini: what Unsloth's llama.cpp adds, and what TensorFold still lacks

- `docs/inkling-2-mini-strata-study.md` reads Unsloth's shipped tree (`b12651-mix-c7059d6`: mainline b12651 plus 43
  pins) against b12651's source: its Inkling gain is seven fused CUDA kernels (conv chain, clip, SiTU, depth mix,
  relative bias in attention, elementwise chains, banded attention), about 7,300 → 3,400 kernels a token. TensorFold's
  Mini engine already fuses all but the loop controller's gate. It marks Unsloth's and Strata's levers in or missing
  with an expected GB10 gain, finds Mia's "Derisked" weights to be a refusal edit at unchanged bytes, and names the
  biggest missing lever (an MXFP8/NVFP4 trunk, +34–58%), not built here because Mini's engine is not on `main`. Inco's
  Splash 1.7.2 joins the list (Unsloth's UD-Q4_K_M weights and a DFlash2 block-diffusion drafter); cards 252–255 carry
  the missing levers.
- `docs/recipes/inkling-2-mini-llamacpp/` builds mainline b12651 and Unsloth's tree on TOP and measures both at 1
  and 4 streams on card 232's UD-Q4_K_M GGUF; its receipt waits on the Desk staging the two source tarballs.
