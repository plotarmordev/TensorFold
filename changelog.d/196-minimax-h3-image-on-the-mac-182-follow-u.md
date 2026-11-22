# MiniMax-H3-Image on the Mac: `--far-pool` only at the gated 4096 x 4096

- `picture.far` binds pooled far-field keys only on `picture.far_grid` (4096 x 4096), the one size card 182's picture
  gates passed at. 4096 x 2048 (8,192 rows) and every other size stay in normal mode with the flag, and
  `tf-h3-image` / `tf-h3-image-server` print `far-pool: off at WxH, not gated`.
- `test-h3-geometry`: 4096 x 4096 pools; 4096 x 2048, 4096 x 4608, 2048, 2048 x 8192 and pictures with references
  stay dense.
