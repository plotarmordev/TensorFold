# MiniMax-H3-Image on Metal: attention scores in registers and 128-row query tiles measured and reverted
- `h3_attention` with its q·kᵀ scores kept in the product's cooperative tensor was tried. Row sums are chained across
  lanes in main's order, and it came out bit-equal to main. 64-row tiles ran 4.89 ms against main's 5.01 ms at the
  2048 shape; 128-row tiles (18,944 threadgroup bytes) ran 5.95 ms. A warm 2048 server A/B gave byte-identical
  latents and PNGs with no visible gain (~36 ms a picture at best), so I reverted the code. The receipts are in the
  Metal recipe's card 162 section.
