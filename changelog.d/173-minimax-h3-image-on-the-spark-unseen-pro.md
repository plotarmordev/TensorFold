# MiniMax-H3-Image on CUDA: the prompt encoder in int8 at a scale per 128 of K, on our own TMA product

- `tf-h3-image-prompt-cuda` runs Qwen3-VL's q/k/v, o, gate/up and down products as int8 with a scale per
  128 of K on both operands, on card 146's TMA kernel in a new grouped mode; `--bf16` keeps the previous
  encoder byte for byte. Below sm_90 the int8 load refuses with `H3Int8RequiresSm90`.
- Same-job A/B on GB10 (job `top-35c84dfda642`): the warm prompt stage on unseen 35–42-token prompts fell
  from 1.9–2.5 s to 0.2 s, resident prompt weights from 48.35 to 26.35 GiB, warm 2048² wall from
  13.9–14.8 s to 12.1–12.2 s. The fox clears episode 1's gates: latent cosine 0.9954, PSNR 29.11 dB,
  mean diff 3.83 (recipe `docs/recipes/minimax-h3-cuda.md`, card 173).
