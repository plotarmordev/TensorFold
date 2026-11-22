# MiniMax-H3-Image on CUDA: opt-in int8 `out` and `fc2` on the TMA product at a scale per 128 of K

`tf-h3-image-cuda --int8-blocks` runs attention out and mlp out on card 146's TMA int8 product too, beside the
default's int8 `qkvc` and `fc1`. A new `h3_quantize_blocks` gives every 128-wide chunk of K its own scale, and
`h3_gemm_int8_tma_blocks` folds each K block's exact int32 sums into fp32 at those scales in registers. It is
0.77 s faster warm at 2048² (13.37 against 14.14 s), but the fox misses episode 1's gates (26.26 dB, mean 5.17),
so the default keeps `out` and `fc2` in bf16. The int8 scratch now holds the widest projection input.
