# MiniMax-H3-Image on CUDA: `--routed-steps N`, FastH3's routed attention in the last steps (opt-in)

- `tf-h3-image-cuda --routed-steps N` routes the last N steps' attention through the video engines' `h3_pool`,
  `h3_tile_scores` and `h3_topk`. Each picture tile keeps the prompt's tiles and its top fifth of the picture's
  tiles. A model without the routing gate now takes `keep` with no pooled branch (it used to be refused with
  `BadH3Ungated`). The default of 0 is main's picture, byte for byte.
- At 2048² on GB10 (job `top-e1cd6974b1da`), routing the last two of six steps cut those steps' attention from
  0.30 s to 0.10 s, plus 0.05 s of routing. The warm picture went from 14.31 s to 13.91 s (2.8%), and every picture
  gate held against the dense picture on the fox and three unseen prompts (`docs/recipes/minimax-h3-cuda.md`).
