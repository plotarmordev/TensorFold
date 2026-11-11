# EXL3 encoder on Mac CPUs

- Add `tf-exl3-encode`, a torch-free Zig CPU encoder for one FP32/FP64 NPY expert matrix and its full calibration Hessian at 2.0, 2.5, 3.0 or 4.0 bits per weight.
- Close every tile's tail-biting trellis: a free beam over the tile turned by half picks the closing state, then a second beam forces the ring's wrapped bits to it. Tile rows run on all cores.
- Give all-zero (pruned) output columns a zero scale, so they decode to exactly zero and add no codebook noise to the live columns of their Hadamard block; reject empty dimensions.
- Write an EXL3/mul1 safetensors group, reload it with the existing reader, verify that `dequantize()` reproduces the encoder's own reconstruction from its chosen states bit for bit, and report EXL3 and NVFP4 Hessian errors with the encode time; `--unweighted` adds the same search without calibration as the same-codebook baseline.
- Add a host NVFP4 quantizer to `core/quant/nvfp4.zig` (ModelOpt scaling: global amax / (6 * 448), E4M3 block-16 scales, E2M1 ties to even) as the encoder's comparator.
- `docs/recipes/exl3.md` records real Drama 3 expert errors beside NVFP4 at each bitrate, seconds per expert and the 5,900-expert projections, with card 32's Hessian format.
