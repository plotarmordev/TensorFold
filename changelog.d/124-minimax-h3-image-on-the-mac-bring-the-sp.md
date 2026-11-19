# 124 — MiniMax-H3 image on the Mac: bring the Spark's wins

Each of the card's four techniques tried on the Metal path; warm 2048 picture, seed 42, six steps, against card 84's
11.14/11.16 s (bar: -10%, ~10.0 s), gates per card 80 (latent cosine >= 0.99, PSNR >= 28.2 dB, mean pixel diff <= 4.1,
byte-identical where arithmetic is unchanged). Full stage numbers and the tried variants are in
`docs/recipes/minimax-h3-image-metal.md` under "Card 124: the Spark's wins on the Metal path".

- **Block-scaled low-precision projections**: int8 qkv at a row's activation scale ran 10.51-10.95 s but broke the
  picture (PSNR 26.16, mean 5.40) — reverted. Activation scales per 256 values like the weights (new kernel
  `h3_gemm_32x64_i8s`) held the gates (PSNR 29.81, cosine 0.9962) at 11.43-11.73 s — the per-256 fold erases the
  int8 win on Metal; qkv alone saved ~0.17 s while its quantize+fold cost more. All four projections int8 grouped
  ran 11.74-13.32 s. Only fc1's row-scaled int8 pays on Metal; kept as on main, everything else reverted.
- **Fused attention**: nothing to port — `h3_attention` is already one fused online-softmax flash kernel; the card's
  stage profile shows it at 1.52-1.54 s of a 9.5 s step (~96 TFLOP/s, the products' own rate).
- **One command buffer a step**: tried as 4-blocks-per-buffer while the speech lock is free, per block once speech
  takes it. No wall gain — a step's 1.5 s is GPU-saturated, so hiding host encode buys nothing; the change also
  broke the latent's byte-identity because a concurrent encoder doesn't order dispatches across blocks and the
  blocks rely on per-buffer commits for the block-boundary dependency. Reverted.
- **Prompt-row work**: `h3_gate_add` and `h3_final_norm` run only on the stepped rows (past `media`) via buffer
  offsets — no kernel change, byte-identical latents (`8b9e75b5…` before and after). Measured saving ~0 (the prompt
  is ~0.6% of rows); kept since it is the technique's port at no cost.

Headline: warm 2048 pictures measured 11.57-11.96 s on the shipped tree — not the 10% bar. The remaining host-side
lever, a GPU-side Euler keeping the latent resident across steps (~0.1 s of the 1.5-1.6 s step wall is wall-GPU gap),
is recorded in the recipe's next-ranked list.
