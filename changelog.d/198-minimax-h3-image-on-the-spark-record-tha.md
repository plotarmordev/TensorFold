# MiniMax-H3-Image on the Spark: bf16 norm tables recorded as no measurable gain

- The CUDA recipe records card 181 (bf16 AdaLN tables, vector loads in `h3_norm_quant`) as measured, no gain shown: byte-identical pictures, warm 2048² median 14.14 s against main's 14.17 s over twelve runs, `norms` 0.62 → 0.60 s. No code changes.
