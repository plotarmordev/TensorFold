# MiniMax-H3-Image on the Spark: GPU decode

- Replace the decoder's cuBLASLt products with ordered FP32 K reductions. Tiled projection reduction order compounded through four products per block into BF16 rounding flips; the sequential order matches the host reference and removes the decoder's cuBLASLt plans and workspace.
- Keep the 6×10 decoder selftest's 90% BF16-identity gate unchanged and add an ordered-projection kernel contract at real reduction depths.
- Spark job `top-52e442a99176` passed every selftest: 6×10 block 1 improved from 86.952% to 97.912% BF16 identity and the larger block 1 reached 93.123%, above the unchanged 90% gate. Its 2048² decode took 115.9 s warm; the same latent through the Metal reference decoder gave 56.77 dB PSNR, 99.934% of channel bytes within 1, and at most 5 apart.
