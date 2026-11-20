# MiniMax-H3-Image on the Spark: the prompt encoder in int8 per-row products, measured and reverted

Qwen3-VL's text layers ran resident as int8 products through main's cuBLASLt int8 path. q/k/v and gate/up each ran
as one product with per-output-row weight scales and per-token activation scales. Each prompt's embedding rows came
from the mapped shard, with the bf16 encoder behind `--bf16`. The same-job A/B (job `top-e69b3bb9e6c2`,
`bench-143-ab.sh` at `4f4731a7`) cut the unseen-prompt stage from 1.22–1.26 s to 0.16–0.17 s, warm 2048² from
15.1–15.2 s to 14.1–14.2 s, and peak memory from 78.6 to 53.8 GB. The prompt states held at cosine ≥ 0.99975 against
the bf16 encoder. But the fox picture missed episode 1's Mac gates: PSNR 27.47 dB and mean diff 4.44, against
28.2 dB and 4.1 (main 28.69 dB, 4.01). Under the card's rule the code is reverted; receipts are in
`docs/recipes/minimax-h3-cuda.md` under card 143.
