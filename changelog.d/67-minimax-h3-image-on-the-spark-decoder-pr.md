# MiniMax-H3-Image CUDA prompt, decoder and AdaLN preparation

- Add CUDA prompt encoding and refinement with streamed BF16 layer weights, plus a batched CUDA VAE decoder. Both expose synthetic comparisons against the existing host references.
- Compute timestep embeddings and AdaLN projections on CUDA, retain block tables across denoising steps, and download only the final host norm's table.
- Select the GPU tools and selftests in the Desk image recipe. CUDA execution, kernel-contract qualification, matched 2048²/4096² pictures and beyond-INT8 timing experiments remain unverified pending a Desk pause window; no new speedup or PSNR is claimed.
