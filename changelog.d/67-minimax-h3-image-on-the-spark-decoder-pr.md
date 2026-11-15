# MiniMax-H3-Image CUDA prompt, decoder and AdaLN preparation

- Add CUDA prompt encoding and refinement with streamed BF16 layer weights, plus a batched CUDA VAE decoder. Both expose synthetic comparisons against the existing host references.
- Compute timestep embeddings and AdaLN projections on CUDA, retain block tables across denoising steps, and download only the final host norm's table.
- Select GPU tools in the Desk recipe; attempt all synthetic selftests before refusing renders on any failure. Preserve the decoder's failed 15 November GPU receipt and the matched mmh3 timings; checkpoint pictures and an end-to-end speedup remain unqualified.
- Preserve the prompt condition projection's bias when selecting a canonical cuBLASLt algorithm, with a host regression for a declined actual-row bias epilogue and an unsupported canonical algorithm.
- Record passing isolated FFN projection and SwiGLU device checks; add cumulative Q/K/V and isolated FP64 attention checks without changing decoder arithmetic or bounds. Decoder qualification and kernel-contract registration remain outstanding.
- Repair the CUDA prompt selftest's missing synthetic embedding tensor and report absent fixture tensors explicitly; exercise the fixture against its written checkpoint on the host. Record the second failed Spark qualification and paired mmh3 timings.
- Record the third failed Spark run: isolated decoder attention passes, while decoder block identity and prompt vision/DeepStack checks fail; neither checkpoint size reaches rendering. Add selftest-only capture of the decoder's intermediate residual to isolate output projections, LayerScale and norm2 without changing production arithmetic or whole-block bounds.
- Keep the failing full prompt vision fixture and add identity-vision, one-layer, splice-only and DeepStack-only comparisons to localize its error without adding production tracing or relaxing numerical gates.
- Record the fourth failed Spark window and paired mmh3 timings without claiming a speedup; isolate decoder norm1, raw QKV and head transforms on their actual GPU inputs while retaining every whole-block gate.
- Initialize the decoder reference scratch worker selector before its first projection; preserve its per-column arithmetic and serial/parallel equality.
- Add selftest-only first-layer prompt captures and same-input FP64/FP32 operator diagnostics, with traced-versus-ordinary byte equality and unchanged whole-run acceptance gates.
- Use the validated actual-row cuBLASLt algorithm when the canonical prompt algorithm is incompatible, preserving required bias and FP32 sums; cover 24-token shapes, rejection modes and cache recovery on the host.
- Record the fifth failed GPU qualification and matched mmh3 timings. Same-input diagnostics do not override whole-block failures; checkpoint PNG parity and GPU stage timings remain outstanding.
