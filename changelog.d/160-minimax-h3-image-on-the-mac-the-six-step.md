# MiniMax-H3-Image on Metal: the GPU Euler is tested against the host's, bit for bit
- `test-h3-image-server` now runs the six steps with the GPU `h3_euler` on the resident latent and with main's old
  host loop (velocity down, `picture.euler`, latent up) on one binding, and requires byte-identical latents. The
  resident Euler itself landed with card 137; no new warm timing, since the render path is unchanged.
