# MiniMax-H3-Image on the Spark: `--fp4`, block-scaled NVFP4 projections on GB10's FP4 tensor cores (opt-in)

`tf-h3-image-cuda --fp4` (`H3_LINEAR=fp4` in the bench scripts) quantizes every transformer projection to e2m1 at
one e4m3 scale every 16 values at load and runs them on cuBLASLt's native block-scaled FP4 products. The microbench
(job `top-7878316ecbe3`) measured 1.98–2.15× the int8 rate on `qkvc`, `out` and `fc1` and 0.90× on `fc2`. In the
same-job A/B (job `top-86442278dd62`), warm 2048² on unseen prompts took 10.6 s against main's 15.1–15.2 s, at
14 GB lower peak memory; the int8 default's latents and PNGs stayed byte for byte main's. The fox picture misses
episode 1's Mac gates (latent cosine 0.8758, 15.28 dB, mean diff 26.48 against 0.99, 28.2 dB and 4.1), with muted
colours, so `--fp4` stays opt-in as a different picture. The H3 model now unloads cuBLASLt after releasing its CUDA
context, whose teardown calls back into the library once FP4 products have run. Receipts and the pictures side by
side are in `docs/recipes/minimax-h3-cuda.md` under card 147.
