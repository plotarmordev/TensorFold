# MiniMax-H3-Image on Metal: the PNG written band by band while the GPU decodes

- The resident server's decoder decodes a canvas-sized picture a tile row a pass. Each finished band of rows goes to
  one streaming zlib `png.Stream` on a CPU thread while the GPU decodes the next tile row. The PNG bytes stay main's
  (`cmp`-identical at 2048 and 4096). In the warm 2048 A/B on the M5 Ultra the decoder stage fell from 0.73-0.78 s to
  0.61-0.66 s, and warm totals averaged 11.46 s against main's 11.54 s over the three least-loaded rounds.
