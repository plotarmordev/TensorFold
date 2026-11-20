# MiniMax-H3-Image: a faster Metal attention, measured and reverted

- At the warm 2048 geometry, `h3_attention` runs at 94 TFLOP/s (5.27 ms a call, about 16% of the six steps). Reducing the softmax row max and sum with simd shuffles, two barriers per key tile instead of three, brought it to 102 TFLOP/s in a microbench. A 32-row fragment kernel in the Spark kernel's shape ran 4x slower.
- End to end on the M5 Ultra (two new prompts, three warm runs each, plus `--draft-steps 3`), the shuffle variant's warm steps matched main within 0.04 s. Its changed reduction order dropped the teapot picture to latent cosine 0.9889 and mean pixel diff 6.38 against main, failing card 80's gate.
- Reverted. The kernel is unchanged, and the receipts are in `docs/recipes/minimax-h3-image-metal.md` under "Card 151".
