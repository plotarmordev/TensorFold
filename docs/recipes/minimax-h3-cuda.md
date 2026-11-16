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
zig build tf-h3-image-prompt-cuda tf-h3-image-cuda tf-h3-image-decoder-cuda -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt-cuda TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42
zig-out/bin/tf-h3-image-cuda MODEL_DIR out/case.safetensors out/case
zig-out/bin/tf-h3-image-decoder-cuda MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
```

On the Sparks, builds and tests run only in GPU jobs on TOP, never through `bottom shell` or `middle shell`: BOTTOM and
MIDDLE serve DeepSeek, and a host build there on 16 November had memguard kill its rank 1 three times. The job image has
no network and no C++ compiler, so [`toolchain-stage.sh`](minimax-h3-cuda/toolchain-stage.sh) first puts the archives
of Zig 0.17.0 and Arm's GNU toolchain 14.3 in TOP's pilot folder; [`cuda-build.sh`](minimax-h3-cuda/cuda-build.sh)
then builds the kernel images with that gcc as nvcc's host compiler, builds `tf-cuda-test` and runs
`kernel-contract-limits` over the five release SMs' images, writing the
images (sm_121 alone for `-Dfatbins`) and the runner to the job output for later TOP jobs:

```sh
ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/cuda-build.sh"
```

| Picture stage | Metal | Spark |
|---|---|---|
| Prompt rows (Qwen3-VL to layer 50, `condition_proj`, the two-block refiner) | `prompt.zig` | CUDA `prompt_cuda.zig`, a layer's weights uploaded at a time; CPU reference `prompt_reference.zig`, the case from `prompt_case.zig` |
| Weights | `model.safetensors.index.json` and its shards | `Model.loadSharded` reads the shards in place |
| Rows in: `video_patch_proj`, F32 [5,376, 96] | `h3_rows_in` | host, `rows_reference.rowsIn` |
| AdaLN tables: the time embedder, each block's projection per timestep | `h3_modulation` | CUDA `cuda_modulation.zig`, resident (line, part, c) tables; final-norm table downloaded |
| Rotary: 48 angles over 96 dims | `picture.rotary`, 96 wide, duplicated | `cos`/`sin` 48 wide, `rot` 48: the first half of each row |
| Transformer blocks, dense, tiles of 64 | `block.zig` | these blocks, `keep` 0 |
| Final norm and the step's head, `pdd_fused_heads` | `h3_final_norm`, `h3_rows_out` | host, `rows_reference.finalNorm`, `rowsOut` |
| Euler step, unpatchify | `picture.euler`, `picture.unpatchify` | the same |
| Image decoder | `image_decoder_metal.zig` | CUDA `image_decoder_cuda.zig`, batched tiles; host blending and PNG output |

Gaps between the video and image transformer: the image transformer has no routing gate (dense attention), fuses each
head's q, k and v, has six per-step output heads instead of one, no audio rows, and its rotary covers 96 of 128 dims.
CUDA and Metal are not bit-identical (SwiGLU rounding, RMS sum order): the check is the picture's PSNR.

`tf-h3-image-prompt-cuda` runs the text encoder and refiner on CUDA, streaming one layer's weights at a time.
`tf-h3-image-decoder-cuda` runs the VAE on CUDA. The CPU tools remain independent reference implementations.
[`run-script.sh`](minimax-h3-cuda/run-script.sh) selects these GPU tools, runs their synthetic selftests and
renders both picture sizes only when they pass. The 14 November tables below predate this cutover; the
[`64188519` window](#gpu-qualification-and-the-64188519-window) measures it.

### First CUDA pictures (BOTTOM, GB10, 14 Nov 2026)

Prompt `h3-image-pilot/prompt.txt` (the red fox, sha256 cc969bca…), seed 42, the checkpoint's six distilled steps,
BF16 throughout, against the M5 Ultra's Metal pictures of card 34 (`h3-image-runs/34/timing`, run 2, same tools):

| Stage | Spark 2048² | Mac 2048² | Spark 4096² | Mac 4096² |
|---|---:|---:|---:|---:|
| Prompt (CPU on the Spark) | 88 s | 17.5 s | 98 s | 14.8 s |
| Transformer load, AdaLN tables on the host | 137 s | 23.3 s | 136 s | 12.5 s |
| Transformer, 6 steps | 34 s (3.84 s a step settled) | 11.1 s (1.85 s) | 123 s (18.5 s) | 59.4 s (9.8 s) |
| Decoder (CPU on the Spark) | 218 s | 2.8 s | 771 s | 34.2 s |
| PSNR against the Mac picture | 29.2 dB | — | 28.5 dB | — |

The Mac's two runs are bit-identical; the CUDA picture differs by the blocks' bf16 rounding (the latents' cosine
0.996). The Metal decoder on the CUDA latent gives the same PSNR at both sizes, so the CPU decoder matches Metal's and
the gap is the transformer's. The first transformer step pays cuBLASLt's algorithm search (4–7 s).

mmh3 ([`9c0652f`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/docs/image.md#speed))
has no BF16 linear layers on CUDA (`--linear-precision int8|nvfp4`, on its own INT8 ConvRot DiT and text encoder).
The matched run therefore keeps everything else: [`mmh3-job.sh`](minimax-h3-cuda/mmh3-job.sh) runs `mmh3 image` on this
prompt, seed 42, 2048² and 4096² and mmh3's defaults, which are the six distilled steps, BF16 attention and no draft steps
(its logs show six). INT8 linear layers, in the text encoder as in the DiT, and its fp16 VAE are the differences.
Its seed draws other noise, so the comparison is of times, not pixels. The job has no network:
[`mmh3-stage.sh`](minimax-h3-cuda/mmh3-stage.sh) first stages mmh3's source, its vendored crates and Rust in the pilot
folder, and the job builds mmh3 from them for `sm_120f`.

The Desk ran it on BOTTOM (GB10) in the 14 Nov 08:05 window, job `bottom-5733d361f24d`; both pictures are clean foxes:

| Stage | TensorFold 2048² | mmh3 2048² | TensorFold 4096² | mmh3 4096² |
|---|---:|---:|---:|---:|
| Prompt | 88 s (CPU) | 3.6 s | 98 s (CPU) | 3.5 s |
| Transformer load | 137 s | 2.9 s | 136 s | 2.8 s |
| Transformer, 6 steps | 34 s (3.84 s a step settled) | 9.5 s (1.4 s) | 123 s (18.5 s) | 53.1 s (8.5 s) |
| Decoder | 218 s (CPU) | 1.0 s | 771 s (CPU) | 2.7 s |
| Wall | 477 s | 17.6 s | 1,128 s | 63.1 s |

On the GPU stage both share, INT8 linear makes mmh3's step 2.7× faster at 2048² and 2.2× at 4096². The rest of the
gap is TensorFold's CPU prompt, host load and CPU decoder, which mmh3 runs on the GPU. mmh3's quoted ~24 s at 4096²
is three PDMD-LoRA steps as half-size drafts plus one full step with INT8/FP8 attention, a different composition.

### GPU qualification and the `64188519` window

Six earlier Desk windows on 15 November failed the synthetic selftests without rendering a checkpoint
picture. The decoder's whole-block identity measured 87.347–92.997% against its 90% bound — worst 6 ULP,
relative error 1.15e-3 — while every operator in the block (norm1, QKV, head norm and rotary, attention,
output projection and LayerScale, norm2, FFN projection, SwiGLU, down projection and LayerScale) passed
its 1-ULP/99%-identical product bound on its own GPU input, and 24 chained blocks sat 1.47e-2 from
float64 against the host fp32 run's 1.5e-2. The prompt's text-only and refiner rows passed; its
vision/DeepStack probes failed the two-times-host gate (4.09e-3 relative error against the host fp32
run's 2.02e-3; a one-layer probe 1.16e-3 against a bit-identical fp32 baseline, which admits none).
Text-to-image never splices vision rows; the real prompt checks against Metal below.

The remaining mismatch was accumulation order. The decoder norms now sum their eight warp partials in the Metal
kernel's serial order and use correctly rounded reciprocal square roots; the image builds with `--fmad=false`.
The text encoder's short-row products now sum K sequentially as the host reference does rather than using a tiled
cuBLASLt reduction, and prompt transcendentals round once from exact. The selftest-only capture entry points are
removed and every numerical bound is unchanged. Whether the repaired kernels pass on device needs a new Desk window;
no fixed-gate result is claimed yet.

The 21 H3 launches' registered usage is nvcc 13.0's, read by `kernel-contract-limits` from fatbins built on BOTTOM
for sm_80/86/89/120/121 at `396df649` (16 November); every launch passes all five SMs' thread, register and shared-memory
limits. The contract's rotary reference now rotates from the unrotated vector under the served cos/sin tables (whose
sines reach ±1), and `h3m_project`'s output is declared and checked as bf16 values in fp32 words.

Desk ran `64188519f690eb8bde9897506e241cdd6e0e4fb8` on TOP's GB10, driver 580.173.02, on 15 November,
18:18–18:29 HKT, with no other job on TOP. `run-script.sh` (`top-43d6798bde4f`) rendered both checkpoint
sizes and exited 1 on the prompt selftest; `mmh3-job.sh` (`top-f6e5bd1bfa56`) exited 0. Every stage ran
on the GPU. The modulation selftest passed; the decoder's block identity measured the same
87.347–92.997%. The script then still rendered after a failure; it now fails fast, so the next window's
pictures require green selftests.

Same-input checks on the M5 Ultra, against this window's files (`h3-image-runs/67/64188519`):

- **Decoder:** the Metal decoder on the GPU run's own latent is 56.58 dB PSNR from the CUDA decoder's 2048² PNG
  (86.0% of bytes identical, 99.9% within 1, at most 7 apart) and 56.19 dB at 4096² (84.8%, 99.9%, at most 6).
- **Prompt:** the Metal prompt tool on the same text gives the same 24 ids and noise; its text rows are cosine
  0.9999558 from the CUDA rows, relative error 9.40e-3, every row at least 0.999886.

Same prompt, seed 42, six steps; one run per size, mmh3's 4096² reusing its 2048² caches. Pictures are both clean
foxes; the seeds draw different noise, so the comparison is of times:

| Stage | TensorFold GPU 2048² | mmh3 2048² | TensorFold GPU 4096² | mmh3 4096² |
|---|---:|---:|---:|---:|
| Prompt | 57.3 s (encoder 55.3 s, refiner 1.6 s) | 3.9 s | 60.1 s | 3.9 s |
| Transformer load, AdaLN tables on the GPU | 88.7 s | 3.2 s | 85.5 s | 3.0 s |
| Transformer, 6 steps | 26.5 s (4.0 s a step settled) | 9.5 s (1.4 s) | 119.6 s (19.5 s) | 55.8 s (8.9 s) |
| Decoder (weights, then tiles) | 7.7 s (4.6 + 3.1) | 1.1 s | 14.9 s (3.8 + 11.1) | 2.8 s |
| Wall | ~180 s | 18.6 s | ~280 s | 66.3 s |

TensorFold's walls sum its four processes' stages. From the 14 November CPU stages this is 477 s to ~180 s at 2048²
and 1,128 s to ~280 s at 4096²; the decoder alone went from 218 s to 7.7 s and 771 s to 14.9 s. mmh3's 17.6 s is
not beaten. Beyond copying mmh3's INT8 path, this card tried a BF16 batched decoder (all tiles in passes on the
GPU, 25 synthetic tiles in 0.55 s, 121 checkpoint tiles in 2.8 s of GPU time) and AdaLN tables built and kept on
the GPU for every step, which the host load no longer computes.

Prompt and load are now weight transport, not arithmetic. The text encoder streams ~45 GiB at ~0.85 GB/s: its
"GPU" time (53.2 s) brackets the layer loop, where each 64 MiB window of the single pinned stage is copied out of the
mapping and waited before the next (`Runner.copy`), and each layer allocates and frees its 11 device buffers. The
load moves 36.7 GiB of blocks the same way (`cuda_weights.upload`, ~0.42 GB/s) and each block's AdaLN projection
weight through pageable copies synced per chunk (`cuda_modulation.Tensor.load`). The transport path now uses two alternating stage windows with mapping read-ahead, reuses each set of layer buffers,
and uploads AdaLN weights through pinned stages. The table remains the previous implementation's baseline; a matched
post-fix benchmark needs a new Desk window. Receipt SHA-256 identifiers:

- `64188519/tensorfold.log`: `5bfda8793e16af8159c8435c098768d26466455c10931db7bd619174262cdb12`
- `64188519/mmh3/out/mmh3/2048.log`: `54e17e0ada53c14502c6dadf543d4f64b0522537e704a383018f78da11772cfb`
- `64188519/mmh3/out/mmh3/4096.log`: `a509701c36d026a9ca502fe904bc723ce73e888a8dd8abc868f800bd72cae3e1`
