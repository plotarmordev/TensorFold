# MiniMax-H3-Image on the Spark: unseen prompts of every length draw again

Prompts of 33–64 and 100+ tokens failed in the CUDA prompt stage with `NoAlgorithm` at condition_proj, the
(rows, 5,376, 5,120) product with its bias fused: when cuBLASLt refused the canonical algorithm at the prompt's
shape, the fallback checked the shape's plain algorithm against a descriptor carrying the bias epilogue. The
fallback now asks the heuristic again with the bias epilogue set, so every length from 1 token to the encoder's
maximum gets an algorithm that adds the bias in its epilogue; the canonical algorithm still runs wherever cuBLASLt
accepts it, so the pilot prompt's latent is unchanged. The transformer's products take the shape's own algorithm the
same way. `tf-h3-image-prompt-cuda` with `H3_PROMPT_CHECK=1` checks each encode's refiner rows against the BF16 host
reference. Receipts are in `docs/recipes/minimax-h3-cuda.md` under card 154.
