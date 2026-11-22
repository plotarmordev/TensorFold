# MiniMax-H3-Image on the Mac: the int8 prompt encoder at a scale per 128 of K, measured and reverted
Card 173's int8 prompt encoder was ported to Metal: q/k/v, o, gate/up and down ran as int8 with a scale per 128 of K on
`h3_gemm_32x64_i8`, and `--bf16` kept main's encoder byte for byte. In a same-session A/B against main `35ff44def`
(one server per side, under `/tmp/tensorfold-model.lock`, the fox cold, then canal, orchard and foundry warm at 2048),
nothing was won. The encoder's GPU time was 0.18 s against 0.29 s one-shot and 0.26-0.34 s against 0.21 s served. It is
about 0.2 s of a 10-12 s warm picture either way. The picture failed card 80's rule against main on all four prompts
(canal 13.1 dB, fox 27.1 dB against the 28.2 dB floor), although the refined rows stayed within cosine 0.9998. The code
is reverted, and the numbers are in `docs/recipes/minimax-h3-image-metal.md` under "Card 192".
