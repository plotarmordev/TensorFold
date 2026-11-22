# MiniMax-H3-Image on the Mac: FastH3's routed attention as an opt-in

`tf-h3-image-server --routed-steps N` routes the last N steps' attention the FastH3 way on the dense picture model:
each picture tile sees the prompt and its best fifth of picture tiles, chosen by the Metal `h3_topk` kernel. It is off
by default, and the default output is unchanged bit for bit. At 2048 two routed steps save about 0.4 s a picture, but
one prompt of three falls 1.6 dB under card 80's 28.2 dB floor (3.3 dB in fast mode), so it is kept as an opt-in,
not a default; the measurements are in `docs/recipes/minimax-h3-image-metal.md` under "Card 187". The tile top-k
contract now builds its library the way the runtime does and serves the 2048 picture's 64-keeping-13 shape.
