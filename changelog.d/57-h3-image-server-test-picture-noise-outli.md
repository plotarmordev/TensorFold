# MiniMax-H3-Image: resident server tests hold steady

- The `test-h3-image-server` cases copy their noise out of the picture file before closing it, so the free run no longer reads unmapped memory and the test binary stops crashing intermittently.
- The failed-bind test allows the device's allocated bytes to sit up to an eighth of one bind's footprint above the baseline, so Metal's buffer rounding no longer fails it, while a leaked bind still does.
- The MiniMax-H3-Image module no longer re-exports the prompt kernels, which nothing used.
