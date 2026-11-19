# MiniMax-H3-Image on the Spark: fold the heads pass into attention, bf16

The dense picture transformer's `h3_heads` pass (norm + rotary over every q/k/v row) runs once a layer for ~0.98 s a
2048² picture. `h3_attention_heads` gathers the fused QKV rows through ROWOF in the tile loads and runs that pass on
the arriving smem tiles — `h3_heads`'s exact bf16 roundings, between the cp.async pipeline's wait and the ldmatrix
reads — so the separate kernel disappears from the dense forward. The routed forward keeps `h3_heads` and the old
attention. Hold only if the warm 2048² picture beats main measured in the same TOP job; the receipt is
`docs/recipes/minimax-h3-cuda/bench-card120.sh` vs `bench-script.sh` at origin/main.
