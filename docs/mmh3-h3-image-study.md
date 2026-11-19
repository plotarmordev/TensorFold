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
GB10, mmh3's warm server ran from [`mmh3-server-job.sh`](recipes/minimax-h3-cuda/mmh3-server-job.sh) (job
`top-a49336186ba9`, this card's `d9fe95b25`): a plain `mmh3 server` with the `cuda,server` build, one cold picture
then three warm repeats of the same prompt and one 1024² request. Ours ran `bench-script.sh` at `9914f752` (job
`top-764629281bea`), the resident servers with `--tc` on the decoder. The stages are mmh3's own log lines and our
bench's SUMMARY columns.

| Stage | TensorFold, warm 2048² | mmh3, warm server 2048² | mmh3, whole run 2048² |
|---|---:|---:|---:|
| Prompt | 1.1 s | 0, the states are kept | 3.4 s encode + 3.1 s encoder load |
| DiT load | 0, resident | 0 | 3.2 s |
| Transformer, six steps | 12.8 s (2.13 s a step) | 8.4 s (1.4 s a step) | 8.4 s |
| Decoder | 1.3 s | 0.6 s | 1.0 s |
| Wall | **15.3 s** | **9.3 s** | **17.2 s** |

Warm means: mmh3 9.31, 9.33, 9.37 s; ours 15.3, 15.3, 15.3 s. The 1024² request took 9.41 s (rendered on a
2048² canvas as `--min-generation-area` sets, Lanczos-resized down). mmh3's steps are flat 1.4 s across all six
sigmas; its first cold step adds 1.1 s of cuBLASLt algorithm search, which the server pays once and caches to
`~/.cache/mmh3`. The kept prompt states skip the encoder's load and encode entirely.

Two things follow. Our step is 1.5× mmh3's on the one stage both run on the GPU, and the remaining 6.0 s of wall
gap is prompt (1.1 s), decoder (0.7 s) and transformer (4.4 s). mmh3's quoted 9.5 s for the second of two 1024²
pictures is close to our measured 9.33 s for three warm 2048² pictures; the 9.5 s figure is confirmed.

## Ranked: what to copy or beat

Expected savings are against our warm 15.5 s (the `40ab38a8` bench), taken from mmh3's measured stage times on the
same Spark, and marked as estimates otherwise.

| # | Change | Saving | Basis and gate |
|---:|---|---:|---|
| 1 | **The transformer's int8 products' own rate**: `qkvc` and `fc1` run at ~94 TFLOPS of int8, a third of the GB10's ~280 dense, and their int32 sums round-trip through `h3_dequantize`; mmh3's ConvRot tile writes BF16 from its own `m16n8k32` mma with no int32 round trip | **−4.4 s** if our products match mmh3's rate | mmh3's 1.4 s a step against our 2.17 s on the same GB10, the transformer's 4.4 s of the 6.2 s wall gap. Our own-tile `h3_gemm_int8` measured 20.5 s warm (reverted, `cdf02b8b4`) — slower than the cuBLASLt it replaced — so the product needs a tile that beats cuBLASLt's heuristic first offer, not a selection change. Keeps the same six steps. Gate: the picture's 28.69 dB and 4.01 apart stay, no bound loosened |
| 2 | **Keep the prompt's states** across pictures, keyed by the prompt | **−1.1 s**, prompt 1.1 → 0 | mmh3's server skips the encoder's load and encode for a repeated prompt (measured: `found the 24 prompt tokens`), ours re-reads the text encoder's ~45 GiB every picture. Keeps the same six steps. Gate: none — the states are identical |
| 3 | **Decoder attention on tensor cores**: `h3d_attention` is 1.88 s of the decoder's 4.54 s GPU time and neither its loads nor its occupancy explain the bound | **−0.5 s** at 2048² if it matches the products' rate | mmh3's decoder is 0.6 s to our 1.3 s. The `--tc` decoder already puts the projections on cuBLASLt; its attention runs as batched q·kᵀ and p·v. The remaining gap is the attention's own tile. Keeps the same six steps. Gate: the decoder's same-latent bound |
| 4 | **Fewer steps**: PDMD's 2-NFE LoRA, or three steps with three half-canvas drafts and one full step through the latent upscaler | **−4.2 s** at three steps (8.4 → 4.2 s after #1) | mmh3's 4096² run falls from 63 s to 24 s with the PDMD LoRA and three drafts, its 2048² one from 18 s to 16 s with two drafts. **Changes the work**: fewer steps and different noise means a different picture — it must pass the same picture gates, and it is an option beside the six-step picture, not a drop-in |
| 5 | **INT8/FP8 attention** | −0.6 s at 2048² (0.1 s a step) | mmh3's 4096² step falls from 8.8 s to 7.3 s, its 2048² one from 1.4 s to 1.3 s. Our `fb5090ae` fp8 round measured 14.1 s a picture (reverted). **Changes the picture**: fine details change, composition holds. Gate: the picture's 28.69 dB and 4.01 |
| 6 | **The final norm, the step's head and Euler on the GPU**, as mmh3 keeps them | ~0.15 s, estimate | Six host round-trips of 4,120 rows replace six GPU passes; the transfer is ~25 MB a step. Do it with #1, whose head is INT8 or FP32 anyway |
| 7 | **Save cuBLASLt's picks and the INT8 tilings** to disk | 4–7 s off the first step of a cold run, 0 warm | The first step pays the algorithm search in every process; a resident server pays it once and caches it |
| 8 | **Threaded in-place checkpoint reads** and two-region streaming | cold start only | mmh3's DiT load is 3.2 s and its loader reads eight threads into place (PR #2). It does nothing for the warm 15.5 s |
| 9 | **NVFP4 linear layers** | ~−1.2 s at 2048², an estimate | mmh3's 4096² step falls from 7.3 s to 6.2 s with muted colors and coarser textures. **Changes the picture**: one mantissa bit is too coarse for the gates |

Numbers 1, 2 and 3 together put a warm 2048² picture at 8.4 + 0.8 + 0 = **9.2 s**, level with mmh3's warm
9.3 s; 4 takes it to about **5.0 s** if the picture gates hold.

## What not to copy

- **Sol-Attn at 2048².** It starts only above 12,288 tokens, and a 2048² picture is 4,120 rows: mmh3's own benchmark
  page says the canvas is too small there and the attention stays dense. It pays at 4096², where we already have
  VSA's routing for video.
- **`--min-generation-area`.** It buys time for requests under 2048² by rendering them at 2048² and resizing down,
  which is a quality choice, not a speed one for our 2048² target.
- **CUDA graphs.** Neither engine uses them; mmh3's launch count is not its problem.
- **Fusing elementwise launches.** The step reads 36.7 GiB of BF16 weights in 2.13 s, about 17 GiB/s against the
  GB10's 273 GB/s, so the step is arithmetic-bound and a fused norm or SwiGLU buys little.

## What card 125 found

mmh3's warm server on TOP's GB10 renders the same prompt at 2048², seed 42, six steps, in **9.33 s** mean
(9.31, 9.33, 9.37), confirming the 9.5 s its docs quote for a 1024² request (which renders on a 2048² canvas).
Our warm 2048² picture is **15.5 s** at `40ab38a8` (prompt 1.1, transformer 13.0, decoder 1.3). The 6.2 s gap
breaks down as: prompt 1.1 s (kept states), decoder 0.7 s, transformer 4.4 s. The transformer is the product
rate — mmh3's ConvRot INT8 tile writes BF16 from its own `m16n8k32` mma with no int32 round trip, while our
cuBLASLt int8 products return int32 sums through `h3_dequantize`. The decoder's attention is its own open
question. nsys is absent from the TOP job image and the pilot folder has 96 GB free, under `put top`'s 100 GB
threshold — the kernel trace waits on a CUPTI preload or a staging path.
