# MiniMax-H3-Image on the Spark: `out` and `fc2` int8 at a scale per 256 of K

A picture's four transformer linears now all run the rotated-int8 cuBLASLt path on CUDA: `qkvc` and `fc1` keep their
per-row scales while `out` and `fc2` carry card 136's finer scales — one per 256 values of K on both operands — the
fold that held card 136's picture gates where card 129's per-row fold failed them (28.15 dB, mean 4.23 against
28.2 dB and 4.1). `h3_quantize` takes a group count (1 is a row's scale, K/256 the grouped one) and a grouped
product is one strided-batched cuBLASLt call — a batch per 256-deep slice of K writing its own (rows, n) int32
plane — whose partials `h3_dequantize_grouped` folds at both operands' per-group scales. `tf-h3-image-cuda`'s
default and `--bf16` switch are unchanged in shape; `--bf16` still runs every projection in bf16.

Card 129's same-shape measurement had all-four int8 0.6 s warm faster than qkvc+fc1 int8 at 2048²; this card's
picture gates and warm A/B against main are in `docs/recipes/minimax-h3-cuda.md` under card 139, job script
`bench-139.sh` (the base side reverse-applies the card's h3 diff inside the job).
