# MiniMax-H3-Image on CUDA: block-scaled int8 and FP8 projections measured and reverted
- GB10's block-scaled FP8 (cuBLASLt MXFP8, e4m3 at a UE8M0 scale every 32 values) was measured and reverted: it
  missed episode 1's gates (latent cosine 0.9664 against 0.99, the picture 11.53 apart from its Mac PNG against
  4.1). The gates are the revert's ground, not speed.
- `h3_gemm_int8`, an int8 `mma.m16n8k32` own tile with a per-128-chunk scale and a bf16 epilogue, ran every
  projection (q k v gate, attention out, mlp in/out) behind `--int8` and passed every 2048² gate on TOP (job
  `top-07d5c772c895`: cosine 0.9959, 29.51 dB and 3.39 apart from the Mac PNG, six steps, decoder `reference` in
  bound). But a warm picture was 20.5 s (transformer 14.4: 3.04, 1.11, 4.52, 2.34) against the cuBLASLt qkvc/fc1
  int8 default's 18.9 s (transformer 12.8), so it is reverted — the own tile is slower on every projection.
