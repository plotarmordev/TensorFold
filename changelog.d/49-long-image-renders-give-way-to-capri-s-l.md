# Image renders give way to live speech on the GPU

- The speech server holds a shared lock on `/tmp/tensorfold-speech.lock` while it synthesizes a request (`zig/src/server/gpu_turn.zig`).
- `tf-h3-image` submits each transformer block as its own command buffer and keeps three queued, so the GPU never idles. Between blocks it waits while any speech holds that lock, so a voice waits at most three blocks (about 0.6 s at 4096 x 4096) instead of a whole 10 s step. It resumes from the same rows: the latent is byte-identical to card 35's at 2048 and 4096, and step times stay within 0.1 s of GPU time at 4096.
- Every step line prints `s waited for speech`. That wait is part of the wall time, so a benchmark leaves out every step where the wait is not 0.
