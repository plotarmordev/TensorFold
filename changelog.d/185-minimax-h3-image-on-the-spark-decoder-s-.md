# MiniMax-H3-Image on the Spark: the PNG written band by band while the GPU decodes

- `tf-h3-image-decoder-cuda` (one-shot and `--serve`) decodes a canvas-sized picture a tile row a pass. Each finished
  band of rows goes to one streaming zlib `png.Stream` on a CPU thread while the GPU decodes the next tile row. A
  resized request keeps the whole-picture path. The PNG bytes stay main's (sha256-identical on four 2048 pictures).
  In the same-job warm 2048 A/B on TOP (`top-b20e9be30f8b`) the decoder's host side fell from 0.40-0.42 s to 0.12 s
  and the decoder stage from 1.18-1.20 s to 1.05 s.
