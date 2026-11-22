# MiniMax-H3-Image on the Mac: products split between the GPU and SME2, measured and stopped

A throwaway microbench under `/tmp/tensorfold-model.lock` split fc2 (5,376 x 14,336) at 4,120 and 16,384 rows. The
GPU ran `h3_gemm_32x64_i8` while twelve SME2 threads took the last rows, on the same int8 operands. The CPU rows
matched the GPU's bf16 output bit for bit, but the best split was 1.015x at 4,120 rows and 1.008x at 16,384, under
the card's 1.07x bar. The GPU product runs at 123-140 TOPS against SME2's 10.4 TOPS bare `smopa` peak and 2.2-3.4
TOPS for a real tile beside the GPU. No engine change: the resident-server split and its A/B were not built. The
table and verdict are in `docs/recipes/minimax-h3-image-metal.md` under "Card 174" and idea 3 of
`docs/recipes/minimax-h3-image-ideas.md`.
