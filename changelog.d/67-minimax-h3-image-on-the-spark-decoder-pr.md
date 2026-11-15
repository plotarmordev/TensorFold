# MiniMax-H3-Image CUDA prompt, decoder and AdaLN preparation

- Add CUDA prompt encoding and refinement with streamed BF16 layer weights, plus a batched CUDA VAE decoder. Both expose synthetic comparisons against the existing host references.
- Compute timestep embeddings and AdaLN projections on CUDA, retain block tables across denoising steps, and download only the final host norm's table.
- Select GPU tools in the Desk recipe; attempt all synthetic selftests before refusing renders on any failure. Preserve the decoder's failed 15 November GPU receipt and the matched mmh3 timings; checkpoint pictures and an end-to-end speedup remain unqualified.
- Preserve the prompt condition projection's bias when selecting a canonical cuBLASLt algorithm, with a host regression for a declined actual-row bias epilogue and an unsupported canonical algorithm.
- Record passing isolated FFN projection and SwiGLU device checks; add cumulative Q/K/V and isolated FP64 attention checks without changing decoder arithmetic or bounds. Decoder qualification and kernel-contract registration remain outstanding.
- Repair the CUDA prompt selftest's missing synthetic embedding tensor and report absent fixture tensors explicitly; exercise the fixture against its written checkpoint on the host. Record the second failed Spark qualification and paired mmh3 timings.
- Record the third failed Spark run: isolated decoder attention passes, while decoder block identity and prompt vision/DeepStack checks fail; neither checkpoint size reaches rendering. Add selftest-only capture of the decoder's intermediate residual to isolate output projections, LayerScale and norm2 without changing production arithmetic or whole-block bounds.
- Keep the failing full prompt vision fixture and add identity-vision, one-layer, splice-only and DeepStack-only comparisons to localize its error without adding production tracing or relaxing numerical gates.
