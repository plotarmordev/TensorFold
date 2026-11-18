# How mmh3 makes a MiniMax-H3-Image image in 17.6 s on a Spark

`mmh3` ([mmh3-org/mmh3](https://github.com/mmh3-org/mmh3), MIT, Rust with CUDA and Metal kernels, no PyTorch or
ComfyUI) makes the same 2048² picture on a DGX Spark in 17.6 s where TensorFold takes 31.4 s warm. This page lists what
it does differently, at [`9c0652f`](https://github.com/mmh3-org/mmh3/tree/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375),
the source of every link below, then ranks what we should copy or beat.

Its own write-up is [`docs/image.md`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/docs/image.md#speed)
and [`docs/performance.md`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/docs/performance.md),
and its posts are the pull-request write-ups [#2](https://github.com/mmh3-org/mmh3/pull/2) (threaded in-place weight
reads), [#3](https://github.com/mmh3-org/mmh3/pull/3) (text encoder fused projections, embedding table on disk),
[#4](https://github.com/mmh3-org/mmh3/pull/4) (batched VAE tiles, FP16 products) and
[#5](https://github.com/mmh3-org/mmh3/pull/5) (VAE read during the steps, kept prompt states), plus
[ryoppippi/mmh3-studio's benchmark page](https://github.com/ryoppippi/mmh3-studio/blob/main/docs/benchmarks/README.md),
which measures mmh3 stage by stage on an M5 Max.

## What it does differently

The TensorFold column is main's H3-Image CUDA path, except where a cell names card 80's branch, whose head carries the
warm 31.4 s picture.

| | mmh3 | TensorFold |
|---|---|---|
| DiT and text encoder weights | INT8 ConvRot from the checkpoint: per-output-row weight scales, per-token-row activation scales taken **after** a normalized Hadamard rotation of each 256-column group; `mma.sync.m16n8k32.s8` products, FP32 scale application, BF16 out ([`int8_gemm.cu:18`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L18), [`convrot.cuh:88`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/convrot.cuh#L88), [`int8_gemm.cu:428`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L428)) | BF16 cuBLASLt Linear with FP32 sums, no quantized H3 path (`zig/src/families/h3/cuda_gemm.zig:1`) |
| Dense attention | FlashAttention-2 style, 128-row query tiles, 64-row K/V tiles, TMA stages, BF16 ([`attention.cu:11`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/attention.cu#L11)) | `h3_attention`: BF16 mma, FP32 online softmax, one tile pair (`zig/kernels/cuda/h3_attention.cuh`) |
| Quantized attention, `--attention-precision int8-fp8` | Q·K in INT8 with per-head, per-64-token-block scales; P·V in FP8 E4M3 with one V scale a head and probabilities scaled by 448 ([`quantized_attention.cu:174`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/quantized_attention.cu#L174)) | none |
| Sparse attention, `--attention sol` | Whole 64-token key blocks a 64-token query block keeps when its pooled score exceeds `tau` × the pooled scores' deviation (tau 1.3), dense for the first 20% of steps and below 12,288 tokens; skipped blocks still enter a pooled softmax term, so softmax spans the image ([`sparse_attention.cu:202`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/sparse_attention.cu#L202), [`dit/sparse.rs:39`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/dit/sparse.rs#L39)) | FastH3's VSA routing kernels exist for video (`h3_pool`, `h3_topk`, `h3_coarse`, `h3_gate_mix`); the picture's dense path uses `keep` 0 |
| Fused layers | One preprocessing launch a block fuses the pending gated residual, the modulated RMSNorm, the AdaLN shift and scale and the ConvRot rotation and quantization; FC1's GEMM writes `silu(gate)·up`; Q/K norm and RoPE fuse the quantize preparation; the GEMM epilogue applies scales, the LoRA adapter, the bias and output maxima ([`dit.rs:1856`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/dit.rs#L1856), [`dit.cu:477`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/dit.cu#L477), [`int8_gemm.cu:511`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L511)) | separate `h3_norm`, `h3_swiglu` and `h3_heads` launches (`zig/src/families/h3/cuda.zig:180`) |
| Steps and output heads | The six sigmas are read from the checkpoint, one FP32 output head a step, Euler; a step off the grid blends the spanned heads by interval length ([`dit/config.rs:47`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/dit/config.rs#L47), [`dit.rs:2880`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/dit.rs#L2880)): the final norm and the head run on the GPU, only the unpatchify on the host | same six sigmas and a head a step, but the final norm, the head, Euler and the unpatchify run on the host every step (`zig/tests/h3_image_cuda.zig:215`) |
| Fewer steps | PDMD's 2-NFE LoRA in two or three steps, and `--draft-steps N` on a half canvas enlarged by a learned latent upscaler or bilinearly, renoised to `--renoise-sigma` ([`image.rs:450`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/image.rs#L450), [`generation.rs:1090`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/generation.rs#L1090)) | six steps only |
| Resolution | `--min-generation-area` 2048²: a smaller request is generated on a 2048² canvas at its aspect ratio and Lanczos-resized after 8-bit quantization ([`image.rs:97`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/image.rs#L97)) | at the requested size |
| Text encoder | INT8 weights, the 1.56 GB embedding table left on disk with only the prompt's rows read, q/k/v and gate/up read as one layer each ([`text_encoder.rs:377`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/text_encoder.rs#L377), PR #3) | BF16, one layer's weights uploaded at a time on main; resident in one device arena on card 80's branch |
| Image VAE | FP16 products through cuBLASLt with FP32 compute; every tile of the canvas in **one** batch, 256-pixel tiles with at least 32 pixels of overlap ([`vae.cu:7`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/vae.cu#L7), [`vae.rs:1324`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/vae.rs#L1324), [`vae.rs:38`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/vae.rs#L38)) | ordered FP32 K sums in 128 × 128 shared tiles, BF16 weights, 16 tiles a batch (`zig/kernels/cuda/h3_image_decoder.cu:131`, `zig/tests/h3_image_decoder_cuda.zig:23`) |
| Caching and residency | A server keeps the encoder, the DiT, both VAEs and the upscaler; the last picture prompt's states are keyed by the encoder's digest, the token ids and the references' digests; `--save-context`/`--context` keep them in a file; the VAE is read while the steps run; refined text states are kept on the GPU ([`resident.rs:105`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/resident.rs#L105), [`image.rs:820`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/image.rs#L820), [`dit.rs:2699`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/dit.rs#L2699)) | resident prompt, transformer and decoder servers (card 80), the prompt re-encoded every picture |
| Algorithm caches | cuBLASLt's picks and the INT8 GEMM's tilings are saved under `~/.cache/mmh3` ([`models.rs:136`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/models.rs#L136)) | cuBLASLt searches each shape in every process (4–7 s of the first step) |
| Load and memory | Eight threads read a checkpoint into place; a model that does not fit keeps what it can and streams the rest through two device regions behind a reader thread ([`streaming.rs:230`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/streaming.rs#L230), [`streaming.rs:165`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/streaming.rs#L165)) | mapped shard reads; the 64188519 window's host load took 88.7 s |
| CUDA graphs | none: no `cudaGraph`, stream capture or persistent whole-model kernel anywhere; only the INT8 GEMM keeps a persistent grid inside its one launch ([`int8_gemm.cu:683`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L683)) | none |

## The same picture both ways on TOP

Both sides ran with `h3-image-pilot/prompt.txt`, seed 42, the checkpoint's six distilled steps and 2048². On TOP's
GB10, mmh3 ran from `mmh3-job.sh` in the Desk's `64188519` window (job `top-f6e5bd1bfa56`, 18.6 s), and ours from
`bench-script.sh` at card 80's head `cecba1009` (job `top-1166fad2c210`, 31.4 s); the stages come from
[`docs/recipes/minimax-h3-cuda.md:284`](recipes/minimax-h3-cuda.md) and card 80's changelog. The same mmh3 job on
BOTTOM's GB10 in the 14 November window took 17.6 s, which is where this card's bar comes from.

| Stage | TensorFold, warm 2048² | mmh3, whole run 2048² | mmh3, warm server |
|---|---:|---:|---:|
| Prompt | 1.1 s | 3.9 s | 0, the states are kept |
| DiT load | 0, resident | 3.2 s | 0 |
| Transformer, six steps | 24.5 s (4.08 s a step) | 9.5 s (1.4 s a step) | 8.4 s |
| Decoder | 5.7 s | 1.1 s | 1.0 s |
| Wall | **31.4 s** | **18.6 s** (17.6 s on BOTTOM's GB10) | **9.5 s** |

Two things follow. Our step is 2.9× mmh3's on the one stage both run on the GPU, and mmh3's whole run pays 7.1 s of
loading and encoding that our warm picture does not, so its **warm** picture is 9.5 s, the number its
[`docs/server.md`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/docs/server.md#asking-for-an-image)
gives for the second of two 1024² pictures (rendered on a 2048² canvas, as `--min-generation-area` sets). 17.6 s is the
bar this card names; 9.5 s is the one to beat.

No fresh single-window run was made in this step: TOP's door refuses `job start gpu` to a builder and allows only
`test`, `test list`, `models status`, `voice`, `jev`, `tell` and `learn`. One Desk window runs both scripts as they
stand, `bench-script.sh` at this head and `mmh3-job.sh`, in about 45 minutes: mmh3's source, its vendored crates and
Rust 1.98 are already staged in TOP's pilot folder as `mmh3-9c0652f`, with the Comfy-Org INT8 ConvRot checkpoint
beside them.

## Ranked: what to copy or beat

Expected savings are against our 31.4 s, taken from mmh3's measured stage times on the same Spark where the two
share a stage, and marked as estimates otherwise.

| # | Change | Saving | Basis and gate |
|---:|---|---:|---|
| 1 | **INT8 ConvRot linear layers** in the DiT, then the text encoder: per-row activations after a 256-column Hadamard rotation, per-output-row weight scales, `m16n8k32` INT8 mma | **−16.1 s**, picture ~15.2 s | mmh3's 1.4 s a step against our 4.08 s on the same GB10, so 24.5 → 8.4 s. Needs the INT8 ConvRot checkpoint (Comfy-Org's, 21 GB) and a rotation-and-quantize kernel; the INT8 mma path already exists for W4A8 (`zig/kernels/cuda/w4a8_prefill.cu:31`). Gate: the picture's 56.65 dB and 3.15e-3 stay, no bound loosened |
| 2 | **FP16 VAE products and every tile in one batch** | **−4.6 s**, decoder ~1.1 s | mmh3's 1.0–1.1 s against our 5.7 s. Our decoder sums K in FP32 in order, which is what the gates were built on; mmh3 runs them in FP16 on both backends; on Metal its pictures are within one level of FP32's everywhere, 68.5 dB at 4096². Gate: the decoder's block identity and the picture's PSNR |
| 3 | **Fewer steps**: PDMD's 2-NFE LoRA, or three steps with three half-canvas drafts and one full step from sigma 0.5 through the latent upscaler | **−4.2 s** at three steps instead of six (8.4 → 4.2 s after #1), and ~2 s more with two half-canvas drafts | mmh3's 4096² run falls from 63 s to 24 s with the PDMD LoRA and three drafts, its 2048² one from 18 s to 16 s with two drafts. The composition changes with the LoRA and the drafts, so this is an option beside the six-step picture, and it must pass the same picture gates |
| 4 | **Keep the prompt's states** across pictures, keyed by the prompt and the references | **−1.1 s**, prompt 1.1 → 0 | mmh3's server skips the encoder's load and encode for a repeated prompt; ours re-reads the text encoder's 45 GiB every picture |
| 5 | **INT8/FP8 attention** | −0.6 s at 2048² (0.1 s a step); −9 s at 4096² | mmh3's 4096² step falls from 8.8 s to 7.3 s, its 2048² one from 1.4 s to 1.3 s. Gate: fine details change, composition holds |
| 6 | **The final norm, the step's head and Euler on the GPU**, as mmh3 keeps them | 0.1–0.5 s, estimate | Six host round-trips of 4,120 rows replace six GPU passes; the transfer is ~25 MB a step, the host pass is not. Do it with #1, whose head is INT8 or FP32 anyway |
| 7 | **Save cuBLASLt's picks and the INT8 tilings** to disk | 4–7 s off the first step of a cold run, 0 warm | The first step pays the algorithm search in every process; a resident server pays it once |
| 8 | **Threaded in-place checkpoint reads** and two-region streaming | cold start only: the 88.7 s host load and the 170.3 s cold wall | mmh3's DiT load is 3.2 s and its loader reads eight threads into place (PR #2). It does nothing for the warm 31.4 s |
| 9 | **NVFP4 linear layers** | ~−1.2 s at 2048², an estimate; −1.1 s a step at 4096² | mmh3's 4096² step falls from 7.3 s to 6.2 s with muted colors and coarser textures. FP8 E4M3 linear layers, which mmh3 has no CUDA path for, are the middle ground to try instead |

Numbers 1, 2 and 4 together put a warm 2048² picture at 8.4 + 1.1 + 0 = **9.5 s**, level with mmh3's warm server and
under the 17.6 s this card names; 3 takes it to about **5.3 s** if the picture gates hold.

## What not to copy

- **Sol-Attn at 2048².** It starts only above 12,288 tokens, and a 2048² picture is 4,120 rows: mmh3's own benchmark
  page says the canvas is too small there and the attention stays dense. It pays at 4096², where we already have
  VSA's routing for video.
- **`--min-generation-area`.** It buys time for requests under 2048² by rendering them at 2048² and resizing down,
  which is a quality choice, not a speed one for our 2048² target.
- **CUDA graphs.** Neither engine uses them; mmh3's launch count is not its problem.
- **Fusing elementwise launches.** The step reads 36.7 GiB of BF16 weights in 4.08 s, about 9 GiB/s against the
  GB10's 273 GB/s, so the step is arithmetic-bound and a fused norm or SwiGLU buys little.

## What card 83 becomes

The study's one large item is INT8 ConvRot linears, and the two small ones beside it are the FP16 VAE and the kept
prompt states; attention precision and NVFP4 matter at 4096² or cost accuracy. So card 83's INT8 round is:

1. INT8 ConvRot linear layers in the DiT, from Comfy-Org's INT8 ConvRot checkpoint, with the picture gates unchanged.
2. The image VAE's products in FP16 with every tile of a canvas in one batch, against the same gates.
3. A warm 2048² picture measured on TOP beside mmh3's own run in one window, at or under 17.6 s, and the 9.5 s of
   mmh3's warm server as the number to beat.
4. Fewer steps as the next round, after those three land.
