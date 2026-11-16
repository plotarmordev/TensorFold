# MiniMax-H3-Image on the Spark: GPU decode

- Replace the decoder's cuBLASLt products with ordered FP32 K reductions. Tiled projection reduction order compounded through four products per block into BF16 rounding flips; the sequential order matches the host reference and removes the decoder's cuBLASLt plans and workspace.
- Keep the 6×10 decoder selftest's 90% BF16-identity gate unchanged and add an ordered-projection kernel contract at real reduction depths.
- Record the 2048×2048 Spark decode timing against the unchanged host float64 reference.
