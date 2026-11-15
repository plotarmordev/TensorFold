# MiniMax-H3-Image CUDA prompt, decoder and AdaLN preparation

- Add CUDA prompt encoding and refinement with streamed BF16 layer weights, plus a batched CUDA VAE decoder. Both expose synthetic comparisons against the existing host references.
- Compute timestep embeddings and AdaLN projections on CUDA, retain block tables across denoising steps, and download only the final host norm's table.
- Select GPU tools in the Desk recipe; attempt all synthetic selftests before refusing renders on any failure. Preserve the decoder's failed 15 November GPU receipt and the matched mmh3 timings; checkpoint pictures and an end-to-end speedup remain unqualified.
- Preserve the prompt condition projection's bias when selecting a canonical cuBLASLt algorithm, with a host regression for a declined actual-row bias epilogue and an unsupported canonical algorithm.
- Add isolated decoder FFN projection and SwiGLU selftest checks on GPU-produced inputs without changing decoding arithmetic or existing tolerances. Their device results and kernel-contract registration remain outstanding.
