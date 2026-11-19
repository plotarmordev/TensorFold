# MiniMax-H3-Image on the Spark: fold the heads pass into attention, bf16 — reverted

Tried folding `h3_heads`' q/k norm and rotary into the dense attention's tile loads: `h3_attention_heads`
gathered the fused QKV rows through ROWOF and ran the pass on the arriving smem tiles with `h3_heads`'s exact
bf16 roundings, so the separate kernel left the dense forward. Correctness held (all selftests PASS, PSNR
56.62 dB, same-latent bound pass, same 6 steps), but the same-job A/B on TOP (job `top-b9b35bcd7d49`,
branch `19a025fa` vs its own checkout reverted to main `a55829e5`) is a clear loss:

| | heads s | attention s | transformer s | warm 2048² s |
|---|---:|---:|---:|---:|
| main | 0.96 | 1.76 | 12.9 | 18.9 |
| fused | 0.00 | 19.94 | 29.8 | 35.9 |

The norm+rotary transform runs once per query-tile × key-tile pair, so its work grows with tiles² instead of
with slots, and it sits inside the mma pipeline between `wait<0>` and `ldmatrix`. The kernel change is
reverted; this receipt keeps the numbers.
