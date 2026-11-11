# MiniMax-H3-Image: weights read ahead

- `tf-h3-image-prompt`, `tf-h3-image` and `tf-h3-image-decoder` read the weight bytes they use into the page cache with 16 parallel 8 MiB reads before Metal touches their in-place mappings, instead of faulting them in one page at a time. A cold picture on an M5 Ultra drops from 229 s to 61 s at 2048x2048 and from 288 s to 123 s at 4096x4096; the case, latent and PNG are byte-identical to before.
- `docs/recipes/minimax-h3-image-metal.md` holds the before-and-after times, the gates, diffusers on MPS and ComfyUI's int8 template measured on the same Mac (a cold 2048 picture: 61 s here, 148 s ComfyUI, 357 s diffusers), and the next speed ideas ranked by measured or reasoned size.
