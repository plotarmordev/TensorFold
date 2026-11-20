# MiniMax-H3-Image on the Spark: per-256-of-K int8 scales on `out`/`fc2` measured and reverted

Card 136's grouped scales (one per 256 values of K on both operands) held the Mac's picture gates where card 129's
per-row scales failed the Spark's. Through cuBLASLt the grouped fold is a strided batch — batch `g` sums the
[g·256, g·256+256) slice of K into its own (rows, n) int32 plane, and a fold kernel scales the planes back — so at
2048² `out` and `fc2` wrote and read ~4 TiB of partial sums a picture. The same-job A/B (job `top-1c32bccda27d`,
`bench-139.sh` at `6bf561216`) measured **34.8 s warm** against main's **14.3 s** (transformer 33.6 vs 13.0), and
the picture gates failed regardless (latent cosine 0.9935, PSNR 28.02 dB, mean 4.27 against 28.2 dB and 4.1). Per
the card's rule the forward path is reverted: `out` and `fc2` stay bf16, `qkvc` and `fc1` keep their per-row int8.
Per-256 scales through the library are closed — only an own-mma tile folding scales in registers can hold both the
gates and the rate; receipts and method in `docs/recipes/minimax-h3-cuda.md` under card 139.
