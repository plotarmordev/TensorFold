# How mmh3 makes a MiniMax-H3-Image image in 9.3 s warm on a Spark

`mmh3` ([mmh3-org/mmh3](https://github.com/mmh3-org/mmh3), MIT, Rust with CUDA and Metal kernels, no PyTorch or
ComfyUI) makes the same 2048² picture on a DGX Spark in 9.3 s warm where TensorFold takes 15.5 s. This page lists what
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

The TensorFold column is main's H3-Image CUDA path at `40ab38a8` (int8 products the default, `--tc` decoder opt-in),
whose warm 15.5 s picture the table below uses.

| | mmh3 | TensorFold |
|---|---|---|
| DiT and text encoder weights | INT8 ConvRot from the checkpoint: per-output-row weight scales, per-token-row activation scales taken **after** a normalized Hadamard rotation of each 256-column group; `mma.sync.m16n8k32.s8` products, FP32 scale application, BF16 out ([`int8_gemm.cu:18`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L18), [`convrot.cuh:88`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/convrot.cuh#L88), [`int8_gemm.cu:428`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L428)) | INT8 cuBLASLt on `qkvc` and `fc1` (per-row absmax, int32 sums through `h3_dequantize`), BF16 on `out` and `fc2` where int8 broke the gates; `h3_gemm_int8`'s own-tile variant measured and reverted (`zig/kernels/cuda/h3_gemm_int8.cu`) |
| Dense attention | FlashAttention-2 style, 128-row query tiles, 64-row K/V tiles, TMA stages, BF16 ([`attention.cu:11`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/attention.cu#L11)) | `h3_attention`: BF16 mma, FP32 online softmax, one tile pair (`zig/kernels/cuda/h3_attention.cuh`) |
| Quantized attention, `--attention-precision int8-fp8` | Q·K in INT8 with per-head, per-64-token-block scales; P·V in FP8 E4M3 with one V scale a head and probabilities scaled by 448 ([`quantized_attention.cu:174`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/quantized_attention.cu#L174)) | none |
| Sparse attention, `--attention sol` | Whole 64-token key blocks a 64-token query block keeps when its pooled score exceeds `tau` × the pooled scores' deviation (tau 1.3), dense for the first 20% of steps and below 12,288 tokens; skipped blocks still enter a pooled softmax term, so softmax spans the image ([`sparse_attention.cu:202`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/sparse_attention.cu#L202), [`dit/sparse.rs:39`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/dit/sparse.rs#L39)) | FastH3's VSA routing kernels exist for video (`h3_pool`, `h3_topk`, `h3_coarse`, `h3_gate_mix`); the picture's dense path uses `keep` 0 |
| Fused layers | One preprocessing launch a block fuses the pending gated residual, the modulated RMSNorm, the AdaLN shift and scale and the ConvRot rotation and quantization; FC1's GEMM writes `silu(gate)·up`; Q/K norm and RoPE fuse the quantize preparation; the GEMM epilogue applies scales, the LoRA adapter, the bias and output maxima ([`dit.rs:1856`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/dit.rs#L1856), [`dit.cu:477`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/dit.cu#L477), [`int8_gemm.cu:511`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L511)) | separate `h3_norm`, `h3_swiglu` and `h3_heads` launches (`zig/src/families/h3/cuda.zig:180`) |
| Steps and output heads | The six sigmas are read from the checkpoint, one FP32 output head a step, Euler; a step off the grid blends the spanned heads by interval length ([`dit/config.rs:47`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/dit/config.rs#L47), [`dit.rs:2880`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/dit.rs#L2880)): the final norm and the head run on the GPU, only the unpatchify on the host | same six sigmas and a head a step; the rows stay on the device since `bd734252` (`h3_rows_in`, `h3_rows_out`, `h3_euler` in `rows_reference`'s order), only the unpatchify on the host |
| Guidance (CFG) | None: the checkpoint is distilled to six sigmas with an output head a step and **no guidance** — a request runs one conditional pass a step, never a batched cond/uncond pair ([`image.rs:6`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/image.rs#L6)) | none: the same distilled checkpoint, same single pass |
| Step or block caching | None: no TeaCache, First-Block-Cache or residual reuse anywhere in the source or the log; the measured steps are flat 1.4 s across all six sigmas — a reused block would show as a cheaper step | none |
| Full resolution | Every step runs the full 4,120-token canvas (the log's `sampling 2048×2048 in 6 steps, 0 at half size`); `--draft-steps` is opt-in only | same: all six steps at full resolution |
| Norms' precision | FP32 accumulations inside the fused `add_norm_quantize`/`rms_norm_modulate` launches (the trace's `PKf...PS2` signatures: f32 states, BF16 weights); the rotate and quantize write int8 on-chip | FP32 states in `h3_norm`, `h3_heads` and `h3_dequantize`'s int32→BF16 pass |
| Compile | No PyTorch, so no `torch.compile`; the Rust host launches every kernel itself | none (Zig host, same) |
| Sigmas | `1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232` read from the checkpoint's `step_sigmas` plus the final, Euler | same six sigmas, same order |
| Fewer steps | PDMD's 2-NFE LoRA in two or three steps, and `--draft-steps N` on a half canvas enlarged by a learned latent upscaler or bilinearly, renoised to `--renoise-sigma` ([`image.rs:450`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/image.rs#L450), [`generation.rs:1090`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/generation.rs#L1090)) | six steps only |
| Resolution | `--min-generation-area` 2048²: a smaller request is generated on a 2048² canvas at its aspect ratio and Lanczos-resized after 8-bit quantization ([`image.rs:97`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/image.rs#L97)) | at the requested size |
| Text encoder | INT8 weights, the 1.56 GB embedding table left on disk with only the prompt's rows read, q/k/v and gate/up read as one layer each ([`text_encoder.rs:377`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/text_encoder.rs#L377), PR #3) | BF16, resident in one device arena; the prompt server re-reads ~45 GiB every picture |
| Image VAE | FP16 products through cuBLASLt with FP32 compute; every tile of the canvas in **one** batch, 256-pixel tiles with at least 32 pixels of overlap ([`vae.cu:7`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/vae.cu#L7), [`vae.rs:1324`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/vae.rs#L1324), [`vae.rs:38`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/vae.rs#L38)) | ordered FP32 K sums in 128 × 128 shared tiles, BF16 weights, 16 tiles a batch (`zig/kernels/cuda/h3_image_decoder.cu:131`, `zig/tests/h3_image_decoder_cuda.zig:23`); `--tc` opts the projections and attention onto cuBLASLt/tensor cores |
| Caching and residency | A server keeps the encoder, the DiT, both VAEs and the upscaler; the last picture prompt's states are keyed by the encoder's digest, the token ids and the references' digests; `--save-context`/`--context` keep them in a file; the VAE is read while the steps run; refined text states are kept on the GPU ([`resident.rs:105`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/resident.rs#L105), [`image.rs:820`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/image.rs#L820), [`dit.rs:2699`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/dit.rs#L2699)) | resident prompt, transformer and decoder servers (card 80), the prompt re-encoded every picture |
| Algorithm caches | cuBLASLt's picks and the INT8 GEMM's tilings are saved under `~/.cache/mmh3` ([`models.rs:136`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/src/models.rs#L136)) | cuBLASLt searches each shape in every process (4–7 s of the first step) |
| Load and memory | Eight threads read a checkpoint into place; a model that does not fit keeps what it can and streams the rest through two device regions behind a reader thread ([`streaming.rs:230`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-core/src/streaming.rs#L230), [`streaming.rs:165`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/src/streaming.rs#L165)) | mapped shard reads; the `64188519` window's host load took 88.7 s, the `40ab38a8` cold wall 155.6 s |
| CUDA graphs | none: no `cudaGraph`, stream capture or persistent whole-model kernel anywhere; only the INT8 GEMM keeps a persistent grid inside its one launch ([`int8_gemm.cu:683`](https://github.com/mmh3-org/mmh3/blob/9c0652f4c39fd67ed2330aeafd61e95a1fe7e375/crates/mmh3-cuda/kernels/int8_gemm.cu#L683)) | none |

## The same picture both ways on TOP

Both sides ran with `h3-image-pilot/prompt.txt`, seed 42, the checkpoint's six distilled steps and 2048². On TOP's
GB10, mmh3's warm server ran from [`mmh3-server-job.sh`](recipes/minimax-h3-cuda/mmh3-server-job.sh) (job
`top-a49336186ba9`, this card's `d9fe95b25`): a plain `mmh3 server` with the `cuda,server` build, one cold picture
then three warm repeats of the same prompt and one 1024² request. Ours ran `bench-script.sh` at `40ab38a8` (job
`top-7cc7074c454e`), the resident servers with `--tc` on the decoder. The stages are mmh3's own log lines and our
bench's SUMMARY columns.

| Stage | TensorFold, warm 2048² | mmh3, warm server 2048² | mmh3, whole run 2048² |
|---|---:|---:|---:|
| Prompt | 1.1 s | 0, the states are kept | 3.4 s encode + 3.1 s encoder load, overlapped |
| DiT load | 0, resident | 0 | 3.2 s |
| Transformer, six steps | 13.0 s (2.17 s a step) | 8.4 s (1.4 s a step) | 8.4 s |
| Decoder | 1.3 s | 0.6 s | 1.0 s |
| Wall | **15.5 s** | **9.3 s** | **17.2 s** |

Warm means: mmh3 9.31, 9.33, 9.37 s; ours 15.5, 15.5, 15.6 s. The 1024² request took 9.41 s (rendered on a
2048² canvas as `--min-generation-area` sets, Lanczos-resized down). mmh3's steps are flat 1.4 s across all six
sigmas; its first cold step adds 1.1 s of cuBLASLt algorithm search, which the server pays once and caches to
`~/.cache/mmh3`. The kept prompt states skip the encoder's load and encode entirely. In the whole-run column
the stages sum past the 17.2 s wall because the encoder load overlaps the encode the request pays for; mmh3's
warm stages sum to 9.0 s against the 9.33 s wall, the rest its request overhead.

Two things follow. Our step is 1.55× mmh3's on the one stage both run on the GPU, and the wall gap's stage
shares are prompt (1.1 s), decoder (0.7 s) and transformer (4.6 s) — 6.4 s, of which ~0.2 s is the same
request/host overhead mmh3 pays. mmh3's quoted 9.5 s for the second of two 1024² pictures is close to our
measured 9.33 s for three warm 2048² pictures; the 9.5 s figure is confirmed.

## Ranked: what to copy or beat

Expected savings are against our warm 15.5 s (the `40ab38a8` bench, job `top-7cc7074c454e`). Ranking rule:
**levers that keep the same six steps first, biggest saving first, then the levers that change the work or the
picture with their gate risk in the last column.**

| # | Change | Saving | Basis and gate |
|---:|---|---:|---|
| 1 | **The transformer's int8 products' own rate and coverage**: `qkvc` and `fc1` run at ~154 TOPS of int8 on cuBLASLt and their int32 sums round-trip through `h3_dequantize`; `out` and `fc2` (286 TFLOP a picture) still run in BF16 at ~84 TFLOPS. mmh3's ConvRot tile writes BF16 from its own `m16n8k32` mma at ~172 TOPS with no int32 round trip, on **all** linears | **−2.2 s** product kernel time (7.74 → 5.55 s) | Same-GB10, same-preload traces (adjacent jobs): mmh3 `int8_gemm` 5.55 s for all 953 TFLOP vs our s8 GEMMs 4.32 s for 667 TFLOP + nvjet BF16 3.42 s for 286 TFLOP. Our own-tile `h3_gemm_int8` measured 20.5 s warm (reverted, `cdf02b8b4`) — slower than the cuBLASLt it replaced — so the product needs a tile that beats cuBLASLt's heuristic first offer, not a selection change. Keeps the same six steps. **Gate risk**: `out` and `fc2` stay BF16 because int8 there broke the gates; matching mmh3 needs its per-token-row ConvRot scales, and the 28.69 dB / 4.01 bounds re-measured |
| 2 | **Fuse the norm+quantize+activation passes** the way mmh3 does: one preprocessing launch a block (gated residual + modulated RMSNorm + AdaLN + ConvRot rotate + quantize), `silu(gate)·up` in the FC1 epilogue, Q/K norm and RoPE folded into quantize preparation | **−1.9 s** (our 3.20 → their 1.34 s) | Same-GB10, same-preload traces: our `h3_dequantize` 0.73 + `h3_swiglu_int8` 0.83 + `h3_norm` 0.51 + `h3_quantize` 0.18 + `h3_heads` 0.95 = 3.20 s of elementwise/quant launches vs mmh3's 1.34 s of fused `add_norm_quantize`, `rotate_quantize`, `qk_norm_rope`. Keeps the same six steps. Gate: rows bit-identical if the fused order keeps the same sums — the FP32 states and BF16 weights already match |
| 3 | **Keep the prompt's states** across pictures, keyed by the prompt | **−1.1 s**, prompt 1.1 → 0 | mmh3's server skips the encoder's load and encode for a repeated prompt (measured: `found the 24 prompt tokens`), ours re-reads the text encoder's ~45 GiB every picture. Keeps the same six steps. Gate: none — the states are identical |
| 4 | **The `--tc` decoder's tile and finish passes**: its 1.04 s warm decode (traced: `h3d_finish` 0.26 s, nvjet+cutlass BF16 products incl. the batched q·kᵀ/p·v attention 0.52 s, `h3d_softmax` 0.08, `h3d_swiglu` 0.08, heads 0.06) against mmh3's 0.55 s VAE | **−0.4 s** at 2048² if it matches mmh3's kernel time | Same-GB10 traces, both engines' decode kernels. The `--tc` decoder already runs `attentionTc` (batched q·kᵀ/p·v inside the nvjet/cutlass GEMMs) — the remaining gap is `h3d_finish` (776 launches of 341 µs) and the products' tile rates, not a missing tensor-core path. Keeps the same six steps. Gate: the decoder's same-latent bound |
| 5 | **The final norm, the step's head and Euler on the GPU**, as mmh3 keeps them | ~0.15 s, estimate | Six host round-trips of 4,120 rows replace six GPU passes; the transfer is ~25 MB a step. Keeps the same six steps. Gate: rows bit-identical — the same sums on device. Do it with #1, whose head is INT8 or FP32 anyway |
| 6 | **Save cuBLASLt's picks and the INT8 tilings** to disk | 4–7 s off the first step of a cold run, 0 warm | The first step pays the algorithm search in every process; a resident server pays it once and caches it. Cold-only, same six steps, no gate |
| 7 | **Threaded in-place checkpoint reads** and two-region streaming | cold start only | mmh3's DiT load is 3.2 s and its loader reads eight threads into place (PR #2). Cold-only, same six steps, no gate; does nothing for the warm 15.5 s |
| 8 | **Fewer steps**: PDMD's 2-NFE LoRA, or three steps with three half-canvas drafts and one full step through the latent upscaler | **−4.2 s** at three steps (~6.4 s transformer after #1+#2) | mmh3's 4096² run falls from 63 s to 24 s with the PDMD LoRA and three drafts, its 2048² one from 18 s to 16 s with two drafts. **Changes the work and the picture**: fewer steps and different noise means a different picture — it must pass the same picture gates, and it is an option beside the six-step picture, not a drop-in |
| 9 | **NVFP4 linear layers** | ~−1.2 s at 2048², an estimate | mmh3's 4096² step falls from 7.3 s to 6.2 s with muted colors and coarser textures. **Changes the picture**: one mantissa bit is too coarse for the gates — the picture's 28.69 dB and 4.01-apart bounds would need re-measuring. Keeps the same six steps but changes the weights |
| 10 | **INT8/FP8 attention** | −0.6 s at 2048² (0.1 s a step) | mmh3's 4096² step falls from 8.8 s to 7.3 s, its 2048² one from 1.4 s to 1.3 s. Our `fb5090ae` fp8 round measured 14.1 s a picture (reverted). **Changes the picture**: fine details change, composition holds. Gate: the picture's 28.69 dB and 4.01. Keeps the same six steps |

Numbers 1+2+3+4 put a warm 2048² picture at 15.5 − 5.6 ≈ **9.9 s**, 0.6 s over mmh3's warm 9.3 s; that remainder
is the wall-vs-kernel host overhead both engines pay between launches, not a stage to drop. #8 on top takes it
to about **5.7 s** if the picture gates hold.

## What not to copy

- **Sol-Attn at 2048².** It starts only above 12,288 tokens, and a 2048² picture is 4,120 rows: mmh3's own benchmark
  page says the canvas is too small there and the attention stays dense. It pays at 4096², where we already have
  VSA's routing for video.
- **`--min-generation-area`.** It buys time for requests under 2048² by rendering them at 2048² and resizing down,
  which is a quality choice, not a speed one for our 2048² target.
- **CUDA graphs.** Neither engine uses them; mmh3's launch count is not its problem.
- **Chasing more launches to cut.** Both traces show the GPU is >99% busy inside each picture — the win is doing
  the same work in fewer, bigger kernels (lever #2), not overlap or graphs.

## What card 125 found

mmh3's warm server on TOP's GB10 renders the same prompt at 2048², seed 42, six steps, in **9.33 s** mean
(9.31, 9.33, 9.37), confirming the 9.5 s its docs quote for a 1024² request (which renders on a 2048² canvas).
Our warm 2048² picture is **15.5 s** at `40ab38a8` (prompt 1.1, transformer 13.0, decoder 1.3). The 6.2 s gap
breaks down as: prompt 1.1 s (kept states), decoder 0.7 s, transformer 4.6 s.

The card asks for nsys; nsys and ncu are absent from the TOP job image, so the profile is a CUPTI preload
([`cupti-trace.c`](recipes/minimax-h3-cuda/cupti-trace.c), LD_PRELOAD, compiled in-job) instead — the deviation
is the tool, not the data: every kernel's name, grid and timestamps. Its full trace holds 11,693 launches over
one cold and two warm pictures; the last warm picture's ~3,300 launches put its kernel budget as:

| Function | Calls | Kernel time | Share |
|---|---:|---:|---:|
| `int8_gemm` (INT8 ConvRot products) | 1,200 | 5.55 s | 63.9% |
| `attention` (dense BF16, 128-row tiles) | 300 | 1.78 s | 20.5% |
| `add_norm_quantize` (fused add+RMS+rotate+quant) | 599 | 0.67 s | 7.7% |
| `qk_norm_rope` | 300 | 0.34 s | 3.9% |
| `rotate_quantize` | 600 | 0.33 s | 3.8% |
| Other (modulation, residual, heads, fill) | ~320 | 0.02 s | 0.1% |

The VAE decode adds 0.55 s of kernel time across ~274 launches (CUTLASS FP16 GEMMs, SwiGLU, residual, norms), of
which 0.62 s is wall. Our side comes from the same kind of trace: `bench-trace-job.sh` ran `bench-script.sh` at
this card's head on the same GB10 under the same `cupti-trace.so` preload — same GPU, same profiler, adjacent
job `top-7abf48024020`, warm mean 15.4 s, transformer 12.8 s. The last warm picture's 3,624 launches, 12.83 s
of kernel time:

| Our kernel | Calls | Kernel time | Role |
|---|---:|---:|---|
| `cutlass_80_tensorop_i16832gemm_s8` | 600 | 4.32 s | int8 `qkvc`+`fc1` products (cuBLASLt) |
| `nvjet_sm121_tst_mma` (BF16) | 600 | 3.42 s | BF16 `out`+`fc2` products |
| `h3_attention` | 300 | 1.75 s | dense BF16 attention |
| `h3_heads` | 300 | 0.95 s | q/k RMSNorm + RoPE |
| `h3_swiglu_int8` | 300 | 0.83 s | SwiGLU on dequantized fc1 |
| `h3_dequantize` | 300 | 0.73 s | int32 sums → BF16 |
| `h3_norm` | 600 | 0.51 s | block norms |
| `h3_quantize` | 600 | 0.18 s | BF16 → int8 inputs |
| `h3_rows_in`/`rows_out`/`gate_add`/`euler` | 24 | 0.15 s | per-step head and Euler |

Achieved rates beside each other on the same GB10 (the DiT is hidden 5376, 56×128 = inner 7168, FFN 14336,
50 blocks, 4,120 rows: ~1,099 TFLOP a picture, ~146 TFLOP of it attention):

| Kernel family | mmh3 | ours | Note |
|---|---:|---:|---|
| int8 products | ~172 TOPS (953 TFLOP, all linears, / 5.55 s) | ~154 TOPS (667 TFLOP on `qkvc`+`fc1` / 4.32 s) | mmh3's own-tile `int8_gemm` is ~12% faster per MAC than our cuBLASLt int8, and covers `out` + `fc2` (286 TFLOP together; `out` is inner 7168→hidden 5376) that ours run in BF16 at ~84 TFLOPS (3.42 s) |
| dense attention | ~82 TFLOPS bf16 (146 TFLOP / 1.78 s) | ~83 TFLOPS bf16 (same / 1.75 s) | identical — same mma shape, same online softmax |
| norm+quant+activation | 1.34 s fused on-chip (`add_norm_quantize`, `rotate_quantize`, `qk_norm_rope`, FC1's `silu(gate)·up` epilogue) | 3.20 s separate (`h3_dequantize`, `h3_heads`, `h3_norm`, `h3_quantize`, `h3_swiglu_int8`) | five launches of int32→BF16→quantize round trips against mmh3's fused preprocessing and GEMM epilogues |

mmh3's transformer does ~1,099 TFLOP in 8.7 s of kernel time; ours the same arithmetic in 12.8 s. The 4.1 s
kernel-time gap is products 2.2 s (rate on the shared int8 work plus the BF16 `out`/`fc2` mmh3 runs in int8)
and the unfused non-GEMM passes 1.9 s.
