# MiniMax-H3-Image on CUDA: the picture's rows on the device

- `tf-h3-image-cuda` keeps the picture's rows on the device across the steps: the rows in, the final norm, the step's head and the Euler update run as `h3_rows_in`, `h3_rows_out` and `h3_euler` in the host reference's sum order, so the latent is unchanged bit for bit and the time outside the blocks falls from 1.2 s to 0.15 s a 2048² picture. `--self-test` checks them against `rows_reference` over two steps.
- With `--int8`, `fc1`'s int32 sums go straight into the SwiGLU and `h3_quantize` reads each row once: a warm 2048² picture on a GB10 takes 18.2 s instead of 19.9 s. `--int8` stays opt-in: its picture is 28.15 dB from episode 1's Mac picture, under the 28.2 dB gate.
