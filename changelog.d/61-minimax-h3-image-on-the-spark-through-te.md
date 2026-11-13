# MiniMax-H3-Image on a Spark, no Metal and no PyTorch

- `tf_h3_open` loads a dense-only H3 checkpoint, MiniMax-H3-Image's `transformer/` blocks, which have no `attn.to_gate_compress`: each block keeps q, k and v alone, its fused per-head q, k, v rows stacked as `h3_heads` reads them. A routed forward (`keep` set) on such a model is refused with `BadH3Ungated`, and setting its gate with `NoGate`.
- `tf-h3-image-prompt-host`, `tf-h3-image-cuda` and `tf-h3-image-decoder-host` make a picture on a Spark: the prompt and the decoder on the CPU through their host oracles, the transformer on the CUDA blocks with the rows in, AdaLN tables, final norm and step heads on the host. Each writes the matching Metal tool's file, so stages swap. `tf-h3-image-cuda` reads the sharded checkpoint in place.
- `docs/recipes/minimax-h3-cuda/run-script.sh` runs the three and the PSNR against the Mac's picture through the GPU door. No CUDA picture has been made yet.
