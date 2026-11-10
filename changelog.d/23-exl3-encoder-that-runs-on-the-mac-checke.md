# EXL3 encoder on Mac CPUs

- Add `tf-exl3-encode`, a torch-free Zig CPU encoder for one FP32/FP64 NPY expert matrix and its full calibration Hessian at 2.0, 2.5, or 3.0 bits per weight.
- Write an EXL3/mul1 safetensors group, reload it with the existing reader, verify bit-exact `dequantize()` reconstruction, and report plain-rounding/NVFP4 Hessian errors and 5,900-expert timing projections.
- Cover circular bit packing, half-bit transitions, weighted-tile fallback, both full Drama 3 expert shapes, and reader/reconstruction equality.
