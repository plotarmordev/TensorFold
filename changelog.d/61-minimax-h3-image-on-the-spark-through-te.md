# MiniMax-H3-Image's blocks on CUDA

- `tf_h3_open` loads a dense-only H3 checkpoint, MiniMax-H3-Image's `transformer/` blocks, which have no `attn.to_gate_compress`: each block keeps q, k and v alone, its fused per-head q, k, v rows stacked as `h3_heads` reads them. A routed forward (`keep` set) on such a model is refused with `BadH3Ungated`, and setting its gate with `NoGate`.
- `docs/recipes/minimax-h3-cuda.md` lists what a text-to-image run on a Spark still needs beyond the blocks: the prompt encoder, rows in, AdaLN tables, final norm and step heads, and the image decoder. No CUDA picture has been made yet.
