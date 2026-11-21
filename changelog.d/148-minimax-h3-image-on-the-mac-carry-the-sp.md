# MiniMax-H3-Image on the Mac: per-kernel GPU counter profile

- `tf-h3-image-server … profile` now times each kernel launch of the six steps in its own encoder between two GPU timestamp counter samples (`mtl.counters`), instead of each stage's command buffer; the stages split into `norm`, `norm quant`, `quantize`, `swiglu`, `gate add`, `head` and `euler`. A profiled latent is the unprofiled one bit for bit (`test-h3-image-server`).
- On an M5 Ultra, card 34's pilot at 2048, warm: the int8 products run at 122-139 TOPS (23-26% of the silicon's 531 TOPS int8) and `h3_attention` at 96 TFLOPS (36% of 265 TFLOPS bf16); card 140's Mac rows are now 10.0 s normal and 6.8 s fast, at load 5-7 and level with main in the same window.
