# MiniMax-H3-Image on CUDA: the prompt encoder resident in int8
- `tf-h3-image-prompt-cuda` now loads Qwen3-VL's text layers as resident int8 products through the int8 product
  path main already runs for the transformer: q, k and v as one product and gate and up as one, each with
  per-output-row weight scales; activations quantize per token before every product (the same Hadamard-128
  rotation h3_quantize gives, streamed by the new `h3p_quantize` so the down product's 25,600-wide rows fit) and
  dequantize by `h3p_dequantize`, which folds o and down's products in as residuals at h3p_add's rounding. The
  embedding table leaves the device: a prompt gathers its own rows from the mapped shard, mmh3-style. The bf16
  encoder stays as `--bf16`; resident weights drop from 46.6 to about 24.4 GiB.
- The unseen-prompt stage measured before the change at 1.07-1.25 s of a warm 2048 picture (side-by-side job of
  card 140); the receipts in `docs/recipes/minimax-h3-cuda.md`'s card-143 section hold the same-job A/B against
  main (three unseen warm runs, `H3_PROFILE` stage times), the prompt states' cosine against the bf16 encoder
  (gate 0.999), the picture gates against episode 1's Mac references, and the device peak with encoder and
  transformer resident.
