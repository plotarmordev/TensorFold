# MiniMax-H3-Image on CUDA: int8 where it pays, and the decoder round measured

- `tf-h3-image-cuda` quantizes `qkvc` and `fc1` and keeps `out` and `fc2` in bf16: the picture passes episode 1's gates
  (28.69 dB and 4.01 apart on average from its Mac PNG, latent cosine 0.9952) where every projection in int8 missed them
  (28.15 dB, 4.23). int8 is the tool's default and `--bf16` runs every projection in bf16.
- The decoder's 128 x 128 tile on 512 threads at 4 x 8 outputs a thread measured slower (products 3.00 s against 2.26 s
  at 256 threads: 12 shared floats read for 32 products against 16 for 64), and `h3d_attention`'s eight 16-byte key
  loads measured neutral (1.89 s against 1.88 s). Both are reverted; the round's numbers are in the recipe.
- A warm 2048² picture on a GB10 is 18.9 s: 1.3 s above mmh3's 17.6 s, with the int8 products' rate (94 TFLOPS) and the
  transformer's attention (217 registers, two CTAs an SM) the rounds left.
