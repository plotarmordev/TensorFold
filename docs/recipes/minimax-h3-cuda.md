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

The block arithmetic is the clip model's; what the picture needs beyond it, read from the Metal tool
(docs/recipes/minimax-h3-image-metal.md), is host work, and none of it is built for CUDA yet:

| Picture stage | Metal | These blocks |
|---|---|---|
| Prompt rows (Qwen3-VL to layer 50, `condition_proj`, the two-block refiner) | `prompt.zig` | none; a host writes `x`'s text rows |
| Weights | `model.safetensors.index.json` and its shards | one file: the host merges the shards' `blocks.*` |
| Rows in: `video_patch_proj`, F32 [5,376, 96] | `rowsIn` | none |
| AdaLN tables: the time embedder on the host, each block's projection per timestep | `modulation.zig` in bf16 | `tables` F32 (blocks, lines, 6, hidden) from the host |
| Rotary: 48 angles over 96 dims | `picture.rotary`, 96 wide, duplicated | `cos`/`sin` 48 wide, `rot` 48: the first half of each row of `picture.rotary` |
| Attention | dense, tiles of 64, prefix first | `keep` 0 with `picture.tiles`' slot and sizes |
| Final norm and the step's head, `pdd_fused_heads` F32 [6, 96, 5,376] | `h3_final_norm`, `rowsOut` | none; the forward ends at the last gated add |
| Euler step, unpatchify | `picture.euler`, `picture.unpatchify` | host, reused as is |
| Image decoder, 2,048 x 2,048 and up | `image_decoder_metal.zig` | none |

A text-to-image run on a Spark therefore still needs a CUDA prompt encoder, the rows-in, table, final-norm and
step-head work, and a CUDA image decoder before the picture, its PSNR against the Mac's and the mmh3 comparison can be
measured. No CUDA picture has been made.
