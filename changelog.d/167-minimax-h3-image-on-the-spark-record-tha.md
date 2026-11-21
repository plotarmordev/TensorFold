# MiniMax-H3-Image on the Spark: the cuBLASLt picks cache, measured with no saving

- `docs/recipes/minimax-h3-cuda.md` records card 141's on-disk cuBLASLt/INT8 pick cache (fork branch head `0ec2c573e`, not merged) as measured, no saving shown: GB10 cold first step 2.07 s without it and 2.08 s with it, the ~1.2 s `warm` figure one pair inside noise, cold wall 152.8 → 149.6 s over one run each way, latents bit-identical. No cache code lands on main.
