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
zig-out/bin/tf-h3-image-decoder-cuda reference MODEL_DIR out/case.latent.f32 128 128 out/case-gpu.png out/case-host.png 2048x2048
```

`reference` decodes the same latent on the GPU and on the host reference decoder, writes both PNGs and compares the
pictures against the selftest's 3e-2 full-chain relative bound.

On the Sparks, builds and tests run only in GPU jobs on TOP, never through `bottom shell` or `middle shell`: BOTTOM and
MIDDLE serve DeepSeek, and a host build there on 16 November had memguard kill its rank 1 three times. The job image has
no network and no C++ compiler, so [`toolchain-stage.sh`](minimax-h3-cuda/toolchain-stage.sh) first puts the archives
of Zig 0.17.0 and Arm's GNU toolchain 14.3 in TOP's pilot folder; [`cuda-build.sh`](minimax-h3-cuda/cuda-build.sh)
then builds the kernel images with that gcc as nvcc's host compiler, builds `tf-cuda-test` and the three H3 picture
tools, and runs `kernel-contract-limits` over the five release SMs' images. It writes the images (sm_121 alone for
`-Dfatbins`), the runner and the picture tools to the job output for later TOP jobs:

```sh
ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/cuda-build.sh"
```

Job `top-7ccced985b0a` ran it at `cf2da2d92`: the build passed and `kernel-contract-limits` exited 1 on 28 stale records
([receipt](../receipts/spark-top-build-cf2da2d92/README.md)).

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
[`run-script.sh`](minimax-h3-cuda/run-script.sh) builds these GPU tools and `tf-cuda-test` with `cuda-build.sh`, runs
their synthetic selftests and renders both picture sizes only when they pass. A failed limits audit does not replace
that selftest gate; its `limits.log` remains in the job output. One TOP job does the build and pictures:

```sh
ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/run-script.sh"
```

The 14 November tables below predate this cutover; the
[`64188519` window](#gpu-qualification-and-the-64188519-window) measures it.

### Benchmark run

[`bench-script.sh`](minimax-h3-cuda/bench-script.sh) builds the sm_121 tools alone in the same TOP job
(`H3_BUILD=sm121 cuda-build.sh`: the five release SMs and their limits audit stay `cuda-build.sh`'s own job, which put
a bench past the Desk's 20 minutes), then takes `run-script.sh`'s inputs and tools without its gate: it records each
selftest's PASS or FAIL and renders anyway. Each size in
`H3_BENCH_SIZES` (default `2048`) gets one cold picture
(start to first picture) and three warm repeats, while `nvidia-smi` is sampled twice a second. The transformer stays
resident: one `tf-h3-image-cuda MODEL --serve` a size loads the model inside the cold run and renders all four cases,
so warm repeats pay no transformer load. `tf-h3-image-prompt-cuda TEXT_ENCODER_DIR MODEL --serve` and
`tf-h3-image-decoder-cuda MODEL --serve` likewise load their weights once a size and serve the four cases. Where unified memory leaves `memory.used` as `[N/A]`, the peak is the
compute processes' memory instead. `/work/out/h3-image-bench/SUMMARY.md` opens with the selftests' verdict. If any
selftest failed or none ran, it says the timings are not the Spark's result. It then lists the commit, GPU, prompt
hash and seed. Per run it gives wall, prompt, load (cold run only), case setup, transformer, decoder load and decoder
(tiles and PNG) seconds, steps, peak MiB, average watts and kJ (average power times wall), plus the warm mean.
[`h3_image_bench.py`](../../tools/zig/h3_image_bench.py) writes it, also when a stage fails. After the resident
servers close, the script decodes the 2048² latent again with `reference`: the GPU's picture and the host reference
decoder's on the same latent, written as PNGs, with the comparison in `2048-reference.compare.log`; the job exits
non-zero past the bound. The host decoder needs the memory the servers held, so it runs once their arenas are free.

    ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-script.sh"

Desk ran `d3ca6199` on TOP's GB10 (driver 580.173.02, job `top-587035ea34ef`, 16 November, 36 minutes, exit 0). The
transformer and prompt selftests passed and the decoder failed, so these timings are not yet the Spark's result:

| Size | Wall cold | Wall warm mean | Prompt | Load | Transformer | Decoder | Peak MiB | Avg W | kJ a picture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048² | 150.4 s | 133.6 s | 21.0 s | 77.6 s | 25.9 s | 3.1 s | 37,994 | 25.6 | 3.4 |
| 4096² | 233.9 s | 237.2 s | 19.8 s | 73.6 s | 125.9 s | 11.1 s | 40,762 | 44.8 | 10.6 |

Stage columns are warm means. In this window each warm repeat still reloaded the model (~75 s) in its own process.
The decoder selftest's failing row is the 6 x 10 case's block 1: 86.952% identical against the 90% bound (worst
3 ULP, relative 1.19e-3); full-size block 1 and the picture bytes pass. This is an open gate failure, the decoder
correctness bug card 72 owns (card 67's run measured 87.3%); the gate stays at 90%. **These timings are not for card
68 until card 72's decoder selftest passes.**

Desk ran `7079fd3b`, the first `--serve` window (job `top-11e9222dd7d6`, 16 November, exit 1), with the same selftest
verdicts. 2048² cold took 185.7 s (load 63.7 s, transformer 25.7 s), but the warm prompt took 88–119 s against
60.6 s cold, and the 4096² decoder died at `cuDevicePrimaryCtxRetain: CUDA_ERROR_OUT_OF_MEMORY` after a 126.9 s
transformer; `d3ca6199` finished it in 11 s. The server kept every transformer shard mapped after the upload: a
second ~38 GB copy of the weights in the GB10's shared memory beside the device's. The prompt tool streams its weights
from the host a layer at a time, so its pages were evicted and re-read mid-run (98.7 s of GPU time warm, 52.3 s
cold), and at 4096² nothing was left for the decoder's context. `--serve` now drops the shards' mapped pages once
the weights are on the device.

Desk ran `8293c6d7` (job `top-72a3cc7eebcc`, 16 November, exit 0), same selftest verdicts, decoder FAIL (card 72).
These timings are a failing build's, **not for card 68**:

| Size | Wall cold | Wall warm mean | Prompt cold / warm | Load | Case setup | Transformer | Decoder | Peak MiB | Avg W | kJ a picture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048² | 159.1 s | 138.9 s | 45.5 / 80.7 s | 51.6 s | 26.1 s | 24.1 s | 3.3 s | 39,576 | 25.7 | 3.6 |
| 4096² | 233.8 s | 261.6 s | 17.4 / 90.2 s | 48.3 s | 38.5 s | 116.6 s | 11.1 s | 42,345 | 40.5 | 10.6 |

Warm columns are means. The 4096² decoder completes again (11.1 s, as at `d3ca6199`, plus 3.6 s of decoder load the
column then left out). The warm prompt was still slow: the text encoder took 69.7–75.7 s of GPU time warm against
37.4 s cold at 2048², and 79.3–87.1 s against 14.0 s at 4096², while the script is serial and the server waits on
stdin, so the prompt was not sharing the GPU's compute. Its event-timed loop includes each layer's host-to-device copy
from the mapped safetensors. Dropping the shards' mappings left their 36.7 GiB in the page cache beside the server's
device copy, so the text encoder's ~45 GiB likely no longer fit in the GB10's shared ~120 GB and was re-read from
disk on every warm prompt; at 4096² cold the previous size's server had exited and freed that room. `--serve` now
also drops the shards from the page cache (`posix_fadvise` `DONTNEED`) once the weights are on the device, and again
after each picture's case setup, which reads the ~26 GB of AdaLN weights back; the next window's warm prompt confirms
or refutes this. Net of the 75 s load it saves, this window's warm wall was no better than `d3ca6199`'s process a
picture (138.9 s against 133.6 s at 2048², 261.6 s against 237.2 s at 4096²). The case setup column is the server's
per-picture preparation before its first step, which `d3ca6199`'s table left inside the wall.

Desk ran `eb8e7312cfa95a943a2c1650d9c319931bacf197` (job `top-fb95a2d49a7a`, 16 November, exit 0). The in-job
`cuda-build.sh` build ran on TOP and wrote `limits.log`; its audit still found the 28 problems card 73 owns. Transformer
and prompt selftests passed and the decoder failed, so these timings remain **not for card 68**. The prompt times were:

| Size | Cold | Warm 1 | Warm 2 | Warm 3 | Warm mean |
|---|---:|---:|---:|---:|---:|
| 2048² | 18.0 s | 22.0 s | 19.3 s | 19.1 s | 20.1 s |
| 4096² | 18.3 s | 18.8 s | 18.1 s | 18.6 s | 18.5 s |

Warm prompt time is back at the cold 18–22 s range at both sizes; the page-cache eviction fixed the warm regression in
this window.

Desk ran `96c87a43` with card 72's ordered FP32 projection kernel (job `top-f42e93268a64`, 16 November, exit 0). Every
selftest passed, so these timings are the Spark's result: at 2048² the decoder took 5.6 s cold and in each warm repeat
(5.25 s on the GPU), and 20.4 s at 4096² (20.3 s warm mean). The same job's `reference` check decoded the 2048² latent
on the GPU and on the host reference decoder: 86.180% of bytes identical, 99.923% within 1, at most 5 apart, 56.65 dB,
relative error 3.15e-3 against the 3e-2 bound, a pass.

At `96c87a43` a warm 2048² picture took 77.0 s: prompt 16.9 s, case setup 26.6 s, transformer 23.6 s (3.94 s a step),
decoder 3.2 s load and 5.5 s tiles. mmh3's is 17.6 s. The case setup was the AdaLN tables, built again for every
picture from ~26 GB of projection weights read back from the shards. Text-to-image's tables depend on the six
timesteps alone, so `--serve` keeps the last picture's tables and rebuilds them only for a case with other
timesteps (a reference's keyframe row). [`serve-check.sh`](minimax-h3-cuda/serve-check.sh) proves the reuse: one
server renders 512² cases a, a, b, a (b adds a 2 x 2 reference, so its table gains the keyframe timestep) and each
latent must equal its case's fresh run byte for byte; the job exits 1 on a miss. [`check-job.sh`](minimax-h3-cuda/check-job.sh)
runs it after the int8 kernels' contract gates, one TOP job. Job `top-de7892d85798` at `1c5074d3` (exit 0) ran
both: the int8 kernels passed gates 1-4 (`h3_quantize`'s eight shapes exact by its own check, `h3_dequantize`'s
within 0.996 of their bound), the live sm_121 limits passed, and served 1-4 matched fresh a, a, b, a byte for byte.

Desk ran that change (`fd812222`, job `top-d795bf3e6c37`, 18 November, exit 0, `bench-script.sh`). Every selftest
passed and the `reference` check repeated `96c87a43`'s numbers (PSNR 56.65 dB, relative error 3.15e-3 against the
3e-2 bound). A warm 2048² picture fell from 77.0 s to **49.7 s**: prompt 16.3 s, case setup 0 s (the tables stay
resident), transformer 24.4 s (4.07 s a step), decoder 1.7 s load and 5.8 s tiles; cold 136.1 s, peak 39.4 GiB,
50.3 W and 2.5 kJ a picture. At 4096² warm is 161.6 s (prompt 18.0, transformer 119.2, decoder 2.2 + 20.6).

Desk ran `d1743eb5` (job `top-5ec3983b311d`, 18 November, exit 0), the resident prompt and decoder servers beside the
transformer's tables. Every selftest passed, but a warm 2048² picture **rose to 68.6 s** (round 1: 49.7 s): prompt
35.8 s, transformer 24.2 s, decoder 2.4 s load and 5.7 s tiles; cold 206.1 s. At 4096² the warm prompt was 9.5 s
against 35.8 s at 2048², so the resident prompt did not cost the ~1 s its weights on the device should.

`h3p_linear` gave every output element its own thread walking that element's weight row: with 24 rows each of a
column's k weights was read 24 times, and neighbouring threads read rows 10 KB apart, so each load used a fraction of
its 32-byte sector. The text encoder's ~45 GiB of weights became ~1.1 TB of traffic, and the memory system's rate,
not the arithmetic, set the stage's time. A warp now owns each output column: its lanes load the column in coalesced
32-element windows into shared memory, and each lane sums one row over them in the reference's k order, so the column
is read once per 32-row group. The dots keep the sequential fp32 sum and every value's rounding, so the text rows are
unchanged.

Desk ran `9cf468df` (job `top-225420da214f`, 18 November) with that kernel, the resident prompt, transformer and
decoder. Every selftest passed, and a warm 2048² picture fell to **35.3 s** (prompt 5.49, transformer 24.13, decoder
5.60); cold 179.1 s, whose prompt server uploads 48.35 GiB in 51.7 s. The job then hung after the 2048 runs in the
`reference` compare: with the three servers holding ~87 GiB, the host decoder thrashed, and the 4096² stage never
started. `bench-script.sh` now runs that compare after the servers close (their arenas freed) and measures 2048² by
default; `H3_BENCH_SIZES='2048 4096'` runs both sizes, which waits on the 4096² memory round.

The prompt's window widened to 256 weights a warp (eight a lane, one 16-byte load), so 512 bytes stay in flight and
eight products follow each load; the dots keep the same sequential order.

Desk ran `f825b43c` (job `top-b16eaa3d03bd`, 18 November) with that window: the prompt fell from 5.48 s to **1.1 s**
warm (45 GiB at 41 GB/s) and a warm 2048² picture from 35.3 s to **31.0 s** (prompt 1.14, transformer 24.21, decoder
5.60; cold 178.3 s). The same job asked cuBLASLt's heuristic at the picture's own 4,120 rows instead of the canonical
16,384-row batch, and the transformer measured 23.91–24.54 s against `9cf468df`'s 24.13 s: the canonical policy stays
and that round is reverted. The job then hung after the 2048 runs, before its `reference` compare and the 4096² stage:
the script's FIFO fds leaked into the resident servers, so the last server held the others' request write ends and
`wait` never saw EOF. `9cf468df`'s ~1.5 h hang was the same leak, not the 4096² stage.

Desk ran `cecba1009` (job `top-1166fad2c210`, 18 November) with the script's fds closed in every server. Every
selftest passed and the job exited 0: a warm 2048² picture took **31.4 s** mean (prompt 1.1, transformer 24.5, decoder
5.7), cold 170.3 s, peak 86.8 GiB, 64.9 W and 2.0 kJ a picture. Its `reference` compare decoded the latent on the GPU
(5.58 s) and on the host (215.6 s) and repeated `96c87a43`'s numbers: 86.180% of bytes identical, 99.923% within 1, at
most 5 apart, PSNR 56.65 dB, relative error 3.15e-3 against the 3e-2 bound, a pass. The host decode is the job's
longest stage after the build, so a round takes ~25 min.

`df74446f` (job `top-de80ff04ca02`, 18 November) ran the transformer with `--int8`: every projection's weight rows and
activation rows Hadamard-rotated in 128-wide chunks, int8 at their absmax, exact int32 sums in cuBLASLt and one bf16
rounding back. Every selftest passed and a warm 2048² picture took **28.9 s** mean (prompt 1.07, transformer 22.2,
decoder 5.6), cold 170.5 s; the latent's cosine against `f825b43c`'s BF16 latent is 0.9961, the BF16 picture's own
cosine against the Mac. But a step only fell from 4.07 s to 3.69 s: the projections were not most of the step on the
GB10, so INT8 alone cannot give mmh3's 1.4 s. The job ran over its 20 min (build and selftests took 20 min of it) and was
stopped in its host `reference` decode; the limits audit fails on the two new unregistered kernels.

`d6fadca4` (job `top-d6126e31f157`) printed the step's profile (`H3_PROFILE`, on in `bench-script.sh`): of a warm
21.8 s transformer, **9.2 s was outside the blocks**, the host's serial `rows_in`, final norm and `rows_out` over
4,096 rows (~1.5 s a step), not attention. In the blocks: mlp in 4.08, q k v gate 2.68, attention 1.76, mlp out 1.72,
heads 0.96, attention out 0.93, norms 0.52 s. `87168278` (job `top-bca977754b29`) splits those host rows over the
cores, each row's sums in the serial order: the latent is byte-identical to `d6fadca4`'s, the transformer takes
13.9 s (1.2 s outside the blocks) and a warm 2048² picture **20.6 s** mean (20.55, 20.55, 20.69: prompt 1.07,
transformer 13.9, decoder 5.6), cold 164.7 s. The job ran 30 min, over its 20 min: the limits audit's fatbin build
took most of it.

`f3f0291f` (job `top-9d030e00d7aa`, 18 November, ~10 min, exit 0) loads the decoder products' K 16 bytes at a time
and prefetches the next 16-deep K slice into registers, with the same ordered sums: the decoder selftest's rows are
unchanged (6 x 10 block 1 97.912%). Every selftest passed and a warm 2048² picture took **20.3 s** mean (20.3, 20.3,
20.4: prompt 1.1, transformer 13.9, decoder 5.3), cold 159.8 s; `reference` passed at 86.044% identical, 99.917% within
1, at most 6 apart, 56.60 dB, relative 3.16e-3. The decoder's GPU time only fell from 5.25 s to 4.95 s, so the product
loads were not its bound; the next decoder round profiles its stages before changing a kernel. The bench now builds
only the sm_121 tools, so the job fits the Desk's 20 minutes.

`b4c5c42` (job `top-c3142e924abc`, 19 November, exit 0) fuses `h3d_linear`'s products into one rounded `__fmaf_rn`
— one rounding a product instead of two, half the instructions — and prints the decoder's per-stage seconds under
`H3_PROFILE`. Every selftest passed; a warm 2048² picture took **19.9 s** mean (19.9, 19.9, 20.0: prompt 1.1,
transformer 13.9, decoder 4.9), cold 161.6 s, peak 71,147 MiB. The decoder's 4.54 s of GPU time split into products
2.26 s, attention 1.88 s, finish 0.25 s, swiglu 0.08 s, heads 0.04 s, norms 0.02 s and embedding 0.00 s: the
products are half of it, and attention is 1.88 s, so both are latency-bound rather than rate-bound. `reference`
repeated its pass at 86.044% of bytes identical, 99.917% within 1, at most 6 apart, PSNR 56.60 dB, relative error
3.16e-3 against the 3e-2 bound. The fma took the decoder's GPU time from 4.95 s to 4.54 s.

`bd734252` (job `top-9190417372e9`, 19 November, exit 0, `bench-script.sh`, `--int8` as the bench's default) sums
`fc1`'s int32 products straight into `h3_swiglu_int8`, quantizes each row from registers after one read, and keeps the
picture's rows on the device (`cuda_rows.zig`: `h3_rows_in`, `h3_rows_out` with the final norm, `h3_euler`, each in
`rows_reference`'s sum order; `--self-test` checks them against it over two steps). Every selftest passed and the
picture is byte for byte `b4c5c42`'s. A warm 2048² picture took **18.2 s** mean (18.1, 18.2, 18.2: prompt 1.1,
transformer 12.1, decoder 4.9), cold 160.7 s; a step is 2.02 s: norms 0.53, q k v gate 2.64, heads 0.96, attention
1.80, attention out 0.94, mlp in 3.34, mlp out 1.78 and 0.15 s outside the blocks (1.2 s before). Against episode 1's
Mac picture (`ep1-mac-fast/2048-warm1.png`, `h3-image-runs/34/t2i-2048.latent.f32`) the `--int8` latent's cosine
is 0.9945, but the picture is 28.15 dB and 4.23 apart on average, past the 28.2 dB and 4.1 gates, so `--int8` stays
opt-in; `b4c5c42`'s 19.9 s was the same `--int8` picture.

`160c5d8b` (job `top-d86e73f6e28d`, 19 November, exit 0) tried two candidates and is reverted: `h3_heads` over a
[tiles, heads] grid kept its stage at 0.99 s (0.96 s before, so the stage is bound by its reads, not its 65 blocks),
and rotating the int8 rows in 256-wide groups as mmh3's ConvRot does left the picture where it was (28.14 dB, 4.23
mean apart, cosine 0.9944). Warm 18.3 s mean (18.2, 18.3, 18.3). The rotation's width is not the picture's gap.

`cb217aa2` (job `top-24813192cb2f`, 19 November, exit 0) ran the decoder's occupancy round and the int8 selection, and
`c30403d9` reverts the decoder half. `h3d_linear` over a 128 x 128 tile on 512 threads at 4 x 8 outputs a thread
(`__launch_bounds__(512, 2)`, ~55 registers, two CTAs and 32 warps an SM, one 16-byte prefetch a thread) measured
products **3.00 s** against 2.26 s at 256 threads: a 4 x 8 thread reads 12 floats of shared memory for 32 products where
8 x 8 reads 16 for 64, so the shared-memory rate, not the warps, bounds the tile. `h3d_attention`'s eight 16-byte key
loads measured 1.89 s against 1.88 s, so the dot's loads were not its bound. The decoder selftest and the same-latent
reference passed on that build (86.117% of bytes identical, 99.916% within 1, at most 5 apart, PSNR 56.62 dB, relative
error 3.16e-3 against the 3e-2 bound). The same job quantized `qkvc` and `fc1` alone: the picture is **28.69 dB** and
**4.01** apart on average from episode 1's Mac PNG with latent cosine 0.9952 (28.15 dB, 4.23, 0.9945 with every
projection in int8), so every gate passes and int8 is now `tf-h3-image-cuda`'s default, `--bf16` opting out. `out` and
`fc2` in bf16 cost 0.72 s of that window (mlp out 1.78 → 2.30, attention out 0.94 → 1.14) and their quantization was
what held the picture at 28.15 dB. `c30403d9` repeated the bench with the decoder reverted (job `top-85cea73cbb20`,
exit 0): every selftest passed, the same-latent check passed at 86.117% of bytes identical and 56.62 dB, the picture
measured the same 28.69 dB, 4.01 apart and 0.9952, and a warm 2048² picture took **18.9 s** mean (18.9, 18.9, 18.9:
prompt 1.1, transformer 12.8, decoder 4.9), cold 157.8 s, peak 76,665 MiB.

Two candidates for the int8 products' round were measured and reverted, each latent byte for byte `c30403d9`'s.
`822b217c` (job `top-be459e1ee06e`) timed the first eight algorithms cuBLASLt's heuristic offers each int8 shape on
the first picture's operands and kept the fastest: q k v gate stayed 2.68 s and mlp in 3.40 s, so the heuristic's
first offer is already the fastest it has. `58857fe9` (job `top-1e5fff43bc3a`) had `h3_heads` read `qkvc`'s int32 sums
and scales and round them itself, skipping `h3_dequantize`'s bf16 rows: q k v gate fell 2.68 → 1.98 s, but heads rose
0.98 → 1.69 s, and the transformer took 12.95–13.05 s against 12.84 s. `h3_heads` already sits at 234 registers and
reads its rows column-scattered, so twice the bytes and a scale a column cost it more than the separate pass did.
Every selftest passed in both windows. A warm picture stays 18.9 s; a product that writes bf16 from its own tile
(a hand-written int8 mma epilogue, not cuBLASLt) is the part of round 1 left. `bdb41d90` (job `top-6a083506b3a5`)
read `h3_heads`' q, k and v rows 16 bytes at a time with each head's q and k staged in 32 KB of shared memory for the
rotary partners, and `h3_swiglu_int8`'s sums and scales four at a time; the latent stayed byte for byte, but heads
rose 0.98 → 2.01 s and mlp in held at 3.39 s (transformer 13.97 s), so the stage's 2-byte loads were not its bound
and the shared rows and two barriers a head cost more. Reverted.

`0c2f142e` re-pipelines `h3_attention` without changing its launch or its arithmetic: the query tile leaves shared
memory for registers (each lane's A fragments read from global once), the freed 16 KB becoming a second value tile so
a key tile's v is fetched two tiles ahead instead of just-in-time, and each mma's b fragment is fetched while the
previous one is in the tensor pipe. The kernel stays bf16 `mma.sync m16n8k16` with an fp32 online softmax, the
softmax's order unchanged, so its output is element for element `c30403d9`'s and every gate is the same. Its TOP
timing (attention seconds a step and the warm 2048² wall, against `c30403d9`'s 18.9 s / 1.80 s attention) is this
card's window; until it runs the stage time is unmeasured.

The rounds left toward mmh3's 17.6 s, in order, each needing its own window:

1. **The int8 products' own rate** is the 1.3 s left: `qkvc` and `fc1` take 2.68 and 3.39 s a picture at ~94 TFLOPS of
   int8, a third of the GB10's dense rate, and their int32 sums make a round trip through `h3_dequantize`. A product
   that writes bf16 straight from the tile would remove that trip, and a per-128-chunk scale would let the products
   carry the picture without the bf16 fallbacks. The two kernels' records are in the registry as of `cadab6b2`, and
   `H3_BUILD=contract sh docs/recipes/minimax-h3-cuda/cuda-build.sh` runs their gates 1-4 (`contract_h3_int8.zig`);
   the limits audit (gate 5) is `limits-job.sh`'s own window.
2. **Attention**: 1.74 s a picture in the transformer, 217 registers at 128 threads and two CTAs an SM (8 warps), and
   1.89 s in the decoder, whose 16-byte key loads changed nothing. mmh3's INT8/FP8 attention is the candidate that
   would also cut registers; the decoder's attention needs its stall found first, since neither its loads nor its
   occupancy explain it.
3. **The rows outside the blocks**: on the device as of `bd734252`, 0.15 s a picture.
4. **The decoder's products** are 2.26 s at 121 registers and two CTAs an SM. A 512-thread 4 x 8 tile measured 3.00 s
   (`cb217aa2`, reverted): the smaller per-thread tile reads 12 shared floats for 32 products where 8 x 8 reads 16 for
   64, so the shared-memory rate rises faster than the warps help. A tile that keeps 8 x 8's ratio at more warps would
   need more registers than a 1024-thread SM allows. Tensor cores stay out: the selftest's 90% block identity
   (86.952% with them, card 72) fails.
5. **The 4096² memory round**: the decoder's buffers do not fit beside the resident prompt (round 2's OOM).

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
[The mmh3 study](../mmh3-h3-image-study.md) lists what mmh3 does differently at its own code, the same picture both
ways, and what each of its tricks would save our warm 31.4 s picture.

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
[`limits-script.sh`](minimax-h3-cuda/limits-script.sh) `[NVCC]` repeats that audit on any host with nvcc, no GPU:
it builds the five SMs' fatbins and `tf-cuda-limits`, then audits the H3 images' records and the controls,
reading the ELF images without opening a driver. `tf-cuda-limits` is `kernel-contract-limits` built without
tf-cuda-test's contracts. At `7079fd3b` the script built all of tf-cuda-test and exited 1 at compile time on BOTTOM,
before any limit ran, in `zig/tests/cuda/contract_drama3_router.zig` (unchanged from main's `c11e8331b`, not this
recipe's code), as Desk reported it: `:100:32 'u8 cannot represent 2312'` and `:169:54 '[]align(1) const i32'`. That
is a main bug for its own card. At `8293c6d7` the scoped script ran on BOTTOM: every H3 launch passes all five SMs'
limits and both controls stay red, but `h3p_norm`, `h3p_heads` and `h3p_attention`'s records were stale (12 rows)
after their sums moved to the host's order; the records now hold that run's measured usage.

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
