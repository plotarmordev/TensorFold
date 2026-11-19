# MiniMax-H3-Image on CUDA: int8 where it pays, and the decoder round measured

- `tf-h3-image-cuda` quantizes `qkvc` and `fc1` and keeps `out` and `fc2` in bf16: the picture passes episode 1's gates
  (28.69 dB and 4.01 apart on average from its Mac PNG, latent cosine 0.9952) where every projection in int8 missed them
  (28.15 dB, 4.23). int8 is the tool's default and `--bf16` runs every projection in bf16.
- The decoder's 128 x 128 tile on 512 threads at 4 x 8 outputs a thread measured slower (products 3.00 s against 2.26 s
  at 256 threads: 12 shared floats read for 32 products against 16 for 64), and `h3d_attention`'s eight 16-byte key
  loads measured neutral (1.89 s against 1.88 s). Both are reverted; the round's numbers are in the recipe.
- A warm 2048² picture on a GB10 is 18.9 s: 1.3 s above mmh3's 17.6 s, with the int8 products' rate (94 TFLOPS) and the
  transformer's attention (217 registers, two CTAs an SM) the rounds left.
- Measured and reverted, latents byte for byte unchanged: timing cuBLASLt's eight offered int8 algorithms a shape
  (no stage moved: the heuristic's first is the fastest), and `h3_heads` reading `qkvc`'s int32 sums itself (q k v
  gate 2.68 → 1.98 s, heads 0.98 → 1.69 s, transformer 12.84 → 12.95 s). The warm picture stays 18.9 s.
- Also measured and reverted: `h3_heads` and `h3_swiglu_int8` reading their rows 16 bytes at a time (heads 0.98 →
  2.01 s, mlp in unchanged at 3.39 s; latent byte for byte unchanged).
