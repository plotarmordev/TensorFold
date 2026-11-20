# MiniMax-H3-Image on the Mac: the int8 product's rate measured in isolation, no kernel change kept

`h3_gemm_32x64_i8` replayed alone at the picture's four shapes (4,120 rows) with command-buffer GPU-timestamp
microbenches (`metal_bench`; isolated launches, not an in-picture profile, which moves to card 148): qkv 138 TOPS,
attention out 125, fc1 139, fc2 122, against measured peaks of 493 TOPS
int8 / 255 TFLOPS bf16 for the register-fed `mma_16x32` chains every product takes — the operand gather binds, not the
tensor unit. Four candidates were microbenched first: int8 fragments as two 4-byte loads (-12 to -19% isolated,
reproduced at load 4.5 and 55-80), a 64x256-threadgroup tile (8-10% slower), register double-buffering and dropping
the per-256 scale fold (both ~70% slower; the fold turns out to break the mma accumulator chain each block). The
4-byte-load candidate passed the latent gates — byte-identical latent (`776b443a…`), card 80's rule against episode 1
(cosine 0.995459, PSNR 28.673 dB, mean diff 3.903), `test-h3-gemm`, `test-h3-image-server`, the kernel contract —
but lost the served A/B on unseen prompts (three warm runs a side, one resident server a side, load reported):
main 8.94/9.25/9.24 s of steps against the candidate's 9.15/9.67/9.67 s. A second A/B in the Desk's GPU window (three
alternated rounds, nine warm runs a side, six steps and `--draft-steps 3`, full stage times and loads recorded)
agreed: median 9.31 s wall / 9.24 s GPU for main against 9.61 s wall / 9.24 s GPU for the candidate (fast mode:
6.09 vs 6.12 s GPU). Kept only if faster: the code was reverted and the receipts, including the micro/served
divergence, are in `docs/recipes/minimax-h3-image-metal.md` under "Card 150".
