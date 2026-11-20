# MiniMax-H3-Image on the Mac: int8 on all four linears

A picture's (the released layout's) four transformer linears — the fused qkv projection, attention out, fc1 and fc2 —
run on `h3_gemm_32x64_i8`, the 32x64-warp int8 product card 109 landed for fc1. Both operands now carry a scale per
256 values of K (card 124's measurements: a row's activation scale breaks the picture at 26.16 dB; the per-256 fold on
both sides held card 80's gates). `bind` quantizes every block's four projections once into private `wq`/`ws` buffers
(each block's refs in `Block.int8`), and each product quantizes its input rows into the shared `xq`/`xs` scratch
between the same stage barriers that already order the block's kernels. `tf-h3-image-server` joins `wq`/`ws` to the
weights' residency set as it did the fc1 copies. A clip's FastH3 blocks make no copies and keep their bf16 products,
held bit for bit by `test-h3-gemm`'s clip case.

A/B on this Mac (card 34's prompt, seed 42, warm 2048, one resident server a side, card 80's gates): latent cosine
0.996884, PSNR 30.613 dB, mean pixel diff 3.451 against main — inside tolerance with more headroom than fc1's
row-scaled landing (28.240 dB). Warm totals 11.18 s against main's 12.50/13.27 s; steps 9.33 s against 9.63/9.64 s.
The bind pays ~5 s once for the 20 GiB copy; the served latent is byte-identical across requests (`776b443a…`,
main's `8b9e75b5…` unchanged). The kernel contract's int8 product now serves all four projection shapes and passes
gates 1-4 on this GPU (worst 0.995 of the float64 bound); the quantizer's served shape is the grouped activation.
`zig build test test-golden` is green apart from `message_starts_proof_test`'s timing bound, which fails identically
on unmodified main. Receipts and method in `docs/recipes/minimax-h3-image-metal.md` under "Card 136".
