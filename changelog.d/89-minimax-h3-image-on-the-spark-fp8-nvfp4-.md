# MiniMax-H3-Image on CUDA: GB10's block-scaled FP8 measured and reverted
- All four transformer projections as cuBLASLt MXFP8 (rotated rows, e4m3 at a UE8M0 scale every 32 values, bf16
  out of the product) miss episode 1's gates on TOP at 2048: latent cosine 0.9664 against 0.99, and the picture
  11.53 apart on average from its Mac PNG against 4.1. The warm picture is also slower, 19.5 s against int8's 18.9 s.
  The code is reverted and int8 `qkvc` + `fc1` stays the default; the recipe has the stage times.
