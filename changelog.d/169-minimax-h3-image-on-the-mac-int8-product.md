# MiniMax-H3-Image on the Mac: the int8 product through threadgroup memory, measured and reverted

`h3_gemm_32x64_i8` with each 32-deep K step's A and B tiles loaded cooperatively (one 16-byte device read a thread a
row half) into double-buffered threadgroup memory, its fragments fed from there (candidate `f9d131dc3`), ran even
with main in isolated replays at 4,120 rows, all output bytes identical: medians of nine paired replays (load
75-128) qkv 8.00 vs 8.32 ms, out 3.47 vs 3.21, fc1 10.75 vs 9.75, fc2 6.97 vs 7.79. 64-deep stages were 37-53%
slower, `matmul2d` straight from the staged tile 23% slower, and the row pitch did not matter. With no isolated
gain, the served A/B was not run and the kernel stays main's; the receipts are in
`docs/recipes/minimax-h3-image-metal.md` under "Card 169".
