# Image renders give way to live speech on the GPU

- The speech server holds a shared lock on `/tmp/tensorfold-speech.lock` while it synthesizes a request (`zig/src/server/gpu_turn.zig`).
- `tf-h3-image` and the `tf-h3-dit` benchmark submit each transformer block as its own command buffer and keep three queued, so the GPU never idles. Between blocks, and once more after the step's output, they wait while any speech holds that lock, so a voice waits at most three blocks instead of a whole 10 s step. A run resumes from the same rows: the latent is byte-identical to card 35's at 2048 and 4096.
- Every step line, and `tf-h3-dit`'s profile, prints `s waited for speech`: exactly 0 when no speech overlapped it, so a benchmark leaves out every timing where it is not 0.
