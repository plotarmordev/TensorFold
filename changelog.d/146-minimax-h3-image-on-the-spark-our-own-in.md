# MiniMax-H3-Image on the Spark: our own int8 TMA product on `qkvc` and `fc1`

- `h3_int8_tma.cuh` ports mmh3's TMA int8 GEMM: a two-stage mbarrier ring, one CTA per SM walking `group_m` bands, `mma.sync.m16n8k32.s8`, and the row and column scales applied in registers before the bf16 store. It runs at 177–187 TOPS against cuBLASLt's 150–156 on all four picture shapes at M = 4,120.
- It replaces cuBLASLt int8 on `qkvc` (bf16 rows straight to `h3_heads`) and `fc1` (SwiGLU in its epilogue over gate and up rows interleaved by `h3_swiglu_rows`). `h3_dequantize`, `h3_heads_i8`, `h3_swiglu_int8` and their int8-only admission rules are gone. Latents are byte-identical to main.
- In the same-job A/B (`bench-146-ab.sh`, three unseen prompts), a warm 2048² picture on the Spark takes 14.2–14.3 s against main's 15.7–15.9 s (transformer 10.9 against 12.3 s). `out` and `fc2` stay bf16. The receipts are in `docs/recipes/minimax-h3-cuda.md`, under card 146.
