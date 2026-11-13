# MiniMax H3 / FastH3 CUDA blocks, one device

This library contains @drowzeys' #958 BF16 transformer blocks and kernel contracts, with the maintainers' boundary,
shared-context, complete-block and allocation repairs. The later shared copy-source owner protects temporary uploads.
Weights and activations retain BF16 storage and products use the runtime's cuBLASLt Linear with FP32 sums.

The host supplies text encoding, sampling, projections, video/audio encoding and decoding. This is a block library,
not a complete native clip-generation endpoint. No weights are included. Code is Apache-2.0; MiniMax H3 / FastH3
weights have the MiniMax H3 Community License and its territorial limits.

Build the real library on a CUDA build machine:

```sh
zig build tf-h3 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast
```

This stage has no rank inputs, transport source, exchange kernel or libibverbs dependency. The C interface is
zig/src/families/h3/cuda_lib.h. Check tf_h3_abi_version() == 1 before passing its 96-byte tf_h3_inputs.
All tiles and rows belong to one device. The library retains the current primary CUDA context through Context.shared;
a host framework's device addresses must belong to that context. If no context is current, the runtime selects zero.

The same entry points support device addresses (tf_h3_forward), host addresses (tf_h3_forward_host), mapped host
allocation, profiling and BF16 projection replacement. The host owns pointer extents and serialized calls on a model.
The fixed 16,384-row cuBLASLt algorithm-selection shape is retained to preserve the original arithmetic policy.

Host tests and compile-only ABI libraries do not load a GPU image:

```sh
zig build test-h3-boundaries test-cuda-host test-cuda-copy-source compile-h3-library
```

Compile-only artifacts are installed under zig-out/host-checks and contain an empty image. They are verification
artifacts, not runnable inference libraries. The Linux library is cross-compiled and is not executed on this Mac.
The numerical contracts, actual kernels, model fidelity, real context interoperability and performance need CUDA hardware.

The original #958 thread reports one/two/three-GB10 results. Those measurements predate this split and do not qualify
this exact source. Sequence-parallel execution is a separate follow-up with an extended ABI and its own transport gates.

The Linux CUDA runner and all H3 contract adapters also have a compile-only gate:

```sh
zig build compile-cuda-tests -Dtarget=x86_64-linux-gnu
```

This step has no NVCC or Metal compiler dependency and does not execute the resulting binary.

## MiniMax-H3-Image on these blocks

MiniMax-H3-Image's `transformer/` uses the clip model's base block names, and the Metal tool reads no
`attn.to_gate_compress` from it (`h3.zig` `baseBlock`): its attention is dense, so no routing gate is needed.
`tf_h3_open` loads such a checkpoint with q, k and v alone in each block's first product (`tf_h3_shape` is
unchanged). The release fuses each head's q, k and v rows (`block.zig`); the load stacks them as `h3_heads` reads
them, all heads' q, then k, then v. A forward with `keep` set is refused with `BadH3Ungated` and
`tf_h3_set_weight` of the gate with `NoGate`, so neither reads past the smaller buffer.

The block arithmetic is the clip model's. Three tools build a picture on a Spark with no Metal and no PyTorch, each
writing the file the matching Metal tool writes (docs/recipes/minimax-h3-image-metal.md), so any stage swaps:

```sh
zig build tf-h3-image-prompt-host tf-h3-image-cuda tf-h3-image-decoder-host -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt-host TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42
zig-out/bin/tf-h3-image-cuda MODEL_DIR out/case.safetensors out/case
zig-out/bin/tf-h3-image-decoder-host MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
```

| Picture stage | Metal | Spark |
|---|---|---|
| Prompt rows (Qwen3-VL to layer 50, `condition_proj`, the two-block refiner) | `prompt.zig` | CPU: `prompt_reference.zig`, a layer's weights in f32 at a time, the case from `prompt_case.zig` |
| Weights | `model.safetensors.index.json` and its shards | `Model.loadSharded` reads the shards in place |
| Rows in: `video_patch_proj`, F32 [5,376, 96] | `h3_rows_in` | host, `rows_reference.rowsIn` |
| AdaLN tables: the time embedder, each block's projection per timestep | `h3_modulation` | host, `rows_reference.modulation`, laid out (line, part, c) as `h3.cu` reads |
| Rotary: 48 angles over 96 dims | `picture.rotary`, 96 wide, duplicated | `cos`/`sin` 48 wide, `rot` 48: the first half of each row |
| Transformer blocks, dense, tiles of 64 | `block.zig` | these blocks, `keep` 0 |
| Final norm and the step's head, `pdd_fused_heads` | `h3_final_norm`, `h3_rows_out` | host, `rows_reference.finalNorm`, `rowsOut` |
| Euler step, unpatchify | `picture.euler`, `picture.unpatchify` | the same |
| Image decoder | `image_decoder_metal.zig` | CPU: `image_decoder_reference.zig` per tile on every core (`image_decoder_host.zig`) |

Gaps between the video and image transformer: the image transformer has no routing gate (dense attention), fuses each
head's q, k and v, has six per-step output heads instead of one, no audio rows, and its rotary covers 96 of 128 dims.
CUDA and Metal are not bit-identical (SwiGLU rounding, RMS sum order): the check is the picture's PSNR.

The prompt encoder and decoder are the bf16 oracles on the CPU, f64 sums: minutes where Metal takes seconds
(prompt 0.47 s, decoder 4.2 s at 2048x2048). mmh3's ~24 s for 4096x4096 compares with the transformer's stage only
until both run on the GPU. [`run-script.sh`](minimax-h3-cuda/run-script.sh) runs all three and the PSNR through the
Desk's GPU door in a DeepSeek pause window. No CUDA picture has been made yet.
