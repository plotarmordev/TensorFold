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
| Rows in: `video_patch_proj`, F32 [5,376, 96] | `h3_rows_in` | `h3_rows_in` (`cuda_rows.zig`, card 122); CPU reference `rows_reference.rowsIn` |
| AdaLN tables: the time embedder, each block's projection per timestep | `h3_modulation` | CUDA `cuda_modulation.zig`, resident (line, part, c) tables; final-norm table downloaded |
| Rotary: 48 angles over 96 dims | `picture.rotary`, 96 wide, duplicated | `cos`/`sin` 48 wide, `rot` 48: the first half of each row |
| Transformer blocks, dense, tiles of 64 | `block.zig` | these blocks, `keep` 0 |
| Final norm and the step's head, `pdd_fused_heads` | `h3_final_norm`, `h3_rows_out` | `h3_rows_out` on the device (final norm + head), `rows_reference`'s order |
| Euler step, unpatchify | `picture.euler`, `picture.unpatchify` | `h3_euler` on the device; `picture.unpatchify` on the host |
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

### Fewer steps: the draft canvas, not the PDMD LoRA

The card's two options were mmh3's PDMD 2-NFE LoRA and its `--draft-steps` half canvas
([the study](../mmh3-h3-image-study.md), lever 8). The LoRA's weights are published
(`yniw/MiniMax-H3-mmh3` on Hugging Face, the step-4000 rank-128 BF16 student) under the
**minimax-h3-community-license-agreement** — a community agreement, not a standard OSS licence; the
job has no network, so they would have to be staged into the pilot folder first, and the adapter would
have to run through the int8 quantized GEMM stack — new surface on top of off-grid head blending.
mmh3 reports the LoRA only at 4096², where its own three-step PDMD runs take about 40 s and its two-step
ones 29 s against 62 s in six, and with three draft steps and the latent upscaler on top the run reaches
24 s; it publishes no 2048² LoRA number. For drafts alone (no LoRA) it reports 2048² at 18 s → 16 s with
two draft steps. Draft steps need no weights: a host bilinear lift plus a second resident stream is the
whole device footprint, and the flag is a direct port of mmh3's own code. At our ~2.15 s a step, three
half-size drafts (~0.6 s each) plus three full steps estimate ~9–10 s of transformer against 12.9 s.

`tf-h3-image-cuda --draft-steps N` ports it: `picture.draftSchedule` is `renoised_schedule`
(first N sigmas at half size, then the renoise — or the next grid sigma — and the undrafted sigmas
below it), `picture.headSpan` is `step_heads` (an off-grid step blends the distilled heads of the
grid intervals its span covers, weighted by interval length; `h3_rows_out` takes an optional
accumulator row and scale so the blend is chained launches, an on-grid step one bare launch), and the
half-canvas draft, its clean prediction and its renoise reuse `cascade.zig`: `cascade.pool` draws the
draft's half-grid noise from the seed's full draw (unit variance, both canvases one picture), and
`cascade.lift` predicts the clean latent `x + σ·v` from the last draft step's video and velocity,
upsamples it bilinearly and renoises it to the next sigma with the seed's own detail. Drafts refuse
references and canvases whose half grid fails `cascade.half`'s 64-pixel rule. `bench-drafts.sh` names
`--draft-steps 3` (with the `--tc` decoder, matching the standing 15.3 s baseline's settings) for the
Desk's env-less door.

`12ce2b1c` (job `top-0aa72d9f154f`, 20 November, exit 0, `bench-drafts.sh` → `run-script.sh` with
`H3_DECODER=tc`, `H3_DRAFT_STEPS=3`; every selftest PASS) ran `--draft-steps 3` beside the `--tc` decoder.
A warm 2048² picture took **10.7 s** mean (10.7, 10.7, 10.7: prompt 1.1, transformer 8.3, decoder 1.3),
cold 148.2 s, peak 76,701 MiB, ~81 W — against the six-step `--tc` baseline's 15.3 s (transformer 12.8),
so the draft cuts a warm picture ~30% at the same decoder. The three draft steps ran ~0.3 s each on the
half canvas, the host lift ~0.1 s, and the three full-size steps ~1.4 s each, so the transformer fell
12.8 → 8.3 s as estimated. mmh3's own `draft-steps 3` server (job `top-508071b72154`, prompt.txt seed 42,
2048²) answered the same request in **6.0 s** warm (5.99, 6.04, 6.02) against its six-step 9.3 s — mmh3
stays leaner end to end; the gap is its pipeline, not the draft math.

**The draft draws a different picture, not a defect — the earlier moiré was a sign bug, now fixed.** A
draft draws a different composition by construction, so the same-picture gate (PSNR ≥ 28.2 dB, mean
diff ≤ 4.1 against the Mac PNG) does not apply; the recipe shows the pictures side by side instead. The
draft's latent cosine against `t2i-2048.latent.f32` is 0.706 (the six-step run is 0.995) — same fox,
different pose and light, because the pooled draft noise and the renoise reshape the layout. The first
build produced a rainbow moiré with no subject (latent cosine −0.11): `Rows.clean` computed `x − σ·v`,
but this engine's velocity points from noise toward data (`v = x₀ − ε`, the opposite of mmh3's
`ε − x₀`), so the clean estimate is `x + σ·v` — the wrong sign folded the latent into ~−0.78·x₀ + 1.78·ε.
Reusing `cascade.lift` (which encodes `x + σ·v` and is covered by `cascade_test`'s exact clean-recovery
case) both fixed the sign and removed the copy-paste `enlargeDraft`/`clean`/`pool` re-implementation;
`enlargeDraft`, the on-device `clean` and the prefix-noise slice are gone.

![`--draft-steps 3` (right) beside the six-step `--tc` picture (left), same prompt and seed 42: a clean
fox in the birch grove in a different pose and light, not the moiré of the first build.](minimax-h3-cuda/draft-vs-six-step.png)

`--draft-steps` stays **opt-in**: it is measurably faster (10.7 s vs 15.3 s warm) and now produces a
clean alternate picture, but it is a different picture, so the lead calls whether it also becomes a
default or recommended mode; the flag is the deliverable either way.

**Warm-serve crash, fixed.** The second `--serve` render segfaulted at `case ready`, while its
`Case.load`/early stream setup ran (job `top-622a63407b13` at `602456336`, exit 139; earlier
`top-85ad023a6cc6`, `top-a09ae34c2ff5`, `top-e83950d1d819` the same). The split of `render` into
`streamSteps` had moved the `folder.tables` call onto the per-request arena, so the kept AdaLN
tables' host memory was freed at the first request's end and the second request's
`std.mem.eql(f32, f.times, times)` guard and `prepare` walked freed pointers — a plain-serve
regression, not the draft math (the crash hit with no `--draft-steps`). `streamSteps` now takes the
persistent `init.gpa` for the cached tables as `render` did on main.

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

`--tc` opts the decoder's products and attention onto tensor cores: each projection runs cuBLASLt
bf16 with fp32 sums into the same `acc` `h3d_finish` reads (the bias and LayerScale keep their
rounding points), and attention runs as batched q·kᵀ and p·v around `h3d_softmax`/`h3d_unheads`.
The tensor-core path is the default from card 122 on; `--exact` opts back into the ordered fp32
kernels, and `bench-script.sh` maps `H3_DECODER=exact` to it.

`2243f9e8` (job `top-2567f9589044`, 19 November, exit 0, `H3_DECODER=tc`) ran the tensor-core decoder beside
the default `52350e5b` window (job `top-b79ca42af474`, exit 0), and `9914f752`'s window (job
`top-764629281bea`, `selftest --tc` PASS) repeated it on the reviewed head: a warm
2048² picture took **15.3 s** mean (15.3, 15.3, 15.3: prompt 1.1, transformer 12.8, decoder 1.3) against the
default's 18.9 s (decoder 4.9), cold 158.4 s. The same-latent `reference` passed at 85.213% of bytes
identical, 99.875% within 1, at most 7 apart, PSNR 56.31 dB (the default's is 56.62) and relative error
3.33e-3, and `tools/zig/h3_image_compare.py` on the fetched `2048-1.png` and latent against episode 1's Mac
picture measured **28.69 dB** and **4.01** apart on average, cosine 0.9952 — the same figures as the
default's. The decoder selftest's staged blocks land ~81–84% identical inside their ULP and relative bounds
— cuBLASLt's fp32 accumulation order, not an error — so `selftest --tc` drops only the identity bar
(`selftest-tc.sh` gates that alone).

Card 130 fuses the `--tc` decoder's finish pass: the bias-free-of-LayerScale projections run
cuBLASLt's bias epilogue — `bf16(acc + b)` in the one rounding `h3d_finish` mode 0 held — and the
LayerScale projections write that same rounded product into `u`, which `h3d_res_norm` folds into
the next block norm (`x = bf16(x + bf16(u * s))`, then the RMS norm). `h3d_finish` had 776 launches
a picture at 341 µs each on the card-125 trace; a CUPTI trace of the same bench
(`top-5a08d40d1ed5`, at `5d2354fa`) shows 48 launches a warm decode — two a block: the qkv (m,3072,1024) and w2
(m,1024,4096) projections, whose heuristic pick (nvjet `tss_128x208`) declines the bias epilogue,
while out, w1 and proj_out run cutlass_80 relu algorithms that take it. Those two shapes keep the
fp32 `acc` and `h3d_finish`. That remainder is cuBLASLt's answer, not a structural one:
`h3d_res_norm` could read w2's fp32 acc and bias directly and `h3d_heads` could add qkv's, a ~0.09 s
fold left for a later card. `--exact` is unchanged. `most_batch` is 128 so a 2048² canvas's 121
tiles decode in one pass, the way mmh3 batches its VAE.

`32279731` (job `top-c328ffe998ad`, 20 November, exit 0) ran the same-job A/B — the bench on this
commit, then again with the card's seven files put back to merge base `60dd5ac5` inside the job.
Every selftest passed in both phases; the same-latent `reference` repeated `9914f752`'s
figures exactly (85.213% of bytes identical, 99.875% within 1, at most 7 apart, PSNR 56.31 dB,
relative 3.33e-3), and every latent and PNG was byte-identical between phases. A warm 2048² picture
took **15.2 s** mean (15.2, 15.2, 15.2: prompt 1.1, transformer 12.8, decoder 1.2–1.3) against the
same job's base phase **15.3 s** (15.3, 15.3, 15.3: decoder 1.3) — kept. The decode's GPU time fell
0.98 → 0.83 s: the finish stage 0.26 → 0.08 s (the 48 remaining `h3d_finish` launches of 0.091 s on
the trace — qkv's nvjet pick and w2's decline the epilogue) and res_norm folded into norms at +0.02 s.
The products' rates are the gap left toward mmh3's 0.55 s decode: products plus attention 0.58 s
(0.34 + 0.24) against the card-125 trace's 0.52 s of nvjet+cutlass GEMM time — the finish pass was
never the products' rate. The trace's per-shape rates: w2 94, w1 81, out 78 TFLOP/s, qkv's nvjet
pick 56, proj_out 42. Against episode 1's Mac picture the fused path's `2048-2` measures the
window's own 28.69 dB and 4.01 apart, cosine 0.9952.

| Warm 2048² | Wall | Prompt | Transformer | Decoder | Same-latent PSNR | vs Mac picture |
|---|---:|---:|---:|---:|---:|---:|
| fp32-order decoder (`52350e5b`, `top-b79ca42af474`) | 18.9 s | 1.1 s | 12.9 s | 4.9 s | 56.62 dB | 28.69 dB, 4.01 |
| tensor-core decoder (`9914f752`, `top-764629281bea`) | 15.3 s | 1.1 s | 12.8 s | 1.3 s | 56.31 dB | 28.69 dB, 4.01 |
| fused finish pass (`32279731`, `top-c328ffe998ad`) | 15.2 s | 1.1 s | 12.8 s | 1.2 s | 56.31 dB | 28.69 dB, 4.01 |

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

`368cee6b` (job `top-309ccf152319`, 19 November, exit 0, `bench-script.sh`) re-pipelined `h3_attention` — the query
tile moved to registers, the freed shared memory double-buffered the value tile a tile ahead, and each mma's b fragment
was fetched ahead of its use — keeping the launch and the softmax's order, so the output was bit-identical. Measured
and reverted (`787f3d3d`): a warm 2048² picture **rose to 19.0 s** mean (19.0, 19.0, 19.0: prompt 1.1, transformer 13.0,
decoder 4.9), cold 161.0 s, a step 2.15 s, attention **1.91 s** a picture — against `c30403d9`'s 18.9 s (18.9, 18.9,
18.9: transformer 12.8), cold 157.8 s, attention **1.80 s**. The decoder's same-latent bound passed (86.117% of bytes
identical, 99.916% within 1, at most 5 apart, PSNR 56.62 dB, relative error 3.16e-3 under 3e-2) and every selftest
passed; the Mac-picture gates (28.2 dB, 4.1, 0.99) are `h3_image_compare.py`'s own run, not this bench's. The regression
is speed, not correctness: the resident query fragments' registers cost more occupancy than the deeper prefetch
recovered — most of the transformer's 0.2 s rise is attention, the rest run noise. Attention's bound is not its value
load's latency.

`acda1e04` (job `top-b76b0efc4246`, 19 November, exit 0, `fp8-attn-bench.sh`) put the dense attention on `mma.sync
m16n8k32` in e4m3 — `h3_attention_fp8`, opt-in on `H3_ATTN=fp8`, staging the raw QKV rows and running `h3_heads`' q/k
RMSNorm and rotary in its prologue so the `h3_heads` pass drops out of the step. A warm 2048²
picture took **39.4 s** mean (transformer 33.3, attention **22.8 s** a picture against `h3_attention` + `h3_heads`' 2.8
s), against the same job's flag-off 19.2 s, and the fp8 latent's cosine against the bf16 latent was **0.8695** with a
15.6 dB picture — selftests passed and the decoder's same-latent bound held (85.643% of bytes identical, 99.897%
within 1, at most 5 apart, PSNR 56.46 dB, relative 3.34e-3) because the selftest does not exercise the flag. The
uniform corruption (per-chunk cosines 0.80–0.89, no masked or permuted region) was per-tile-absmax e4m3 being too
coarse on Q·K — four significand bits spread over a whole tile.

`fb5090ae` (job `top-ea3a4d434c59`, 19 November, exit 0, same script) moved q and k to int8 at each row's absmax —
the mmh3 split, e4m3 kept only on the weighed-score × value product — and prefetched each key tile's stage during
the previous tile's value product: the warm 2048² picture came to **30.3 s** (attention **14.1 s**) with the latent
cosine **0.9882** and a 24.5 dB picture against the same job's flag-off 18.9 s. Both gates still fail: 0.988 is
under the 0.99 cosine and 30.3 s over the 18.9 s. Reverted in full: what is left is not tuning — the serialized
stage→quantize→mma prologue costs ~11 s a picture against `h3_attention`'s pipelined tile loop, so on the GB10 the
scale granularity and the bf16↔fp8 conversion cost more than the narrower mma returns at this tile size, and the
last 0.002 of cosine likely needs the weighed scores in int8 too rather than e4m3. A follow-up that pipelines the
prologue and finishes the precision split starts from these measurements.

`19a025fa` (job `top-b9b35bcd7d49`, 19 November, exit 0, same-job A/B against its own checkout reverted to
`a55829e5`) folded `h3_heads`' q/k norm and rotary into `h3_attention_heads`' tile loads: the loads gathered the
fused QKV rows through ROWOF, and each arriving q/k smem tile took `h3_heads`'s exact bf16 roundings between the
pipeline's wait and the ldmatrix reads, so `h3_heads` left the dense forward. Every selftest passed and the
same-latent bound held, but the attention stage **rose 1.76 → 19.94 s** a picture (heads 0.96 → 0.00, transformer
12.9 → 29.8 s, warm mean 18.9 → 35.9 s): the transform runs once per query-tile × key-tile pair, so its work
grows with tiles² instead of slots, and it sits in the mma pipeline's critical path. Reverted (`a560c4eb`); the
heads pass's own 0.98 s is what a folded variant would have to beat, and inside attention the per-pair rewrite
costs far more. The once-per-row alternative — norm and rotary in the qkvc product's epilogue instead of per
tile pair — was not measured; it is the remaining shape for folding the heads pass into attention.
GB10's block-scaled FP8 was measured and reverted (card 89). `160c7687` (job `top-ad7ecf144cee`, 19 November) ran
all four projections as cuBLASLt MXFP8: each 128-wide chunk rotated as the int8 rows are, then e4m3 at one UE8M0
scale every 32 values, the product writing bf16 from its own tile. Its device check of a 200 x 256 x 384 product
with a 64x outlier column came within 3.85e-2 relative error of float64, e4m3's three mantissa bits on each
operand. The transformer selftest failed — the run's log ends in a segfault — and its SUMMARY records a failing
build, so that run's stage timings are not cited; the script still rendered, and the picture missed every gate:
latent cosine 0.9664 (gate 0.99) and 11.53 apart on average from the Mac PNG (gate 4.1), the three warm latents
byte for byte the same. The failed selftest and the picture gates are the revert's ground. NVFP4 keeps one
mantissa bit where e4m3 keeps three and is expected to miss the gates further; it has not been measured.

`626fd8ded` put each denoising step in one CUDA graph — rows in, every block's launches, the step head
and the Euler update — replayed on a dedicated stream (`H3_PROFILE`, which waits after each stage, keeps
the eager path), staged every step's AdaLN lines once and pointed each step's graph at its own slice;
`--serve` restores the model's own stream when a picture returns. The `--tc` decoder became the default
in the same window, `--exact` keeping the fp32-order kernels and their identity selftest. A decoder-side
rows overlap was also built and measured — the last step streaming its video rows out band by band while
a progressive `.rows.f32` reader unpatchified, denormalised and uploaded each band and launched a tile
pass once the rows its tiles read had landed — and job `top-27eefb4ad8de` measured it slower at 2048²:
the banded download + per-band event waits added ~0.9 s to the last step and the progressive host decode
cost 3.6 s vs 1.3 s for the bulk path (warm 15.7 s), so it was cut and the decoder keeps its `.latent.f32`
input. The keep-if-faster A/B first ran as two jobs on the same GPU that day (the job clone is too
shallow for an in-checkout git revert): `top-c0b33c78c637` (`bench-122-ab.sh` at `ec0aa8c84`, exit 128 —
its before phase's `git checkout` failed on the shallow clone, but its after phase ran to a finished
summary) measured the candidate's warm mean 15.2 s (transformer 12.8, ~2.15 a step from graph replay,
decoder 1.3), and `top-7cc7074c454e` (`bench-script.sh` at `40ab38a8f` on scratch branch
`scratch/122-baseline`, the card's h3 files reverted to the merge base with the decoder forced `--tc`)
measured the baseline's 15.5 s (transformer 13.0, steps ~2.16, decoder 1.3) — 0.3 s faster, kept. Every
selftest passed in both runs, the same-latent `reference` bound held at 56.31 dB (rel 3.33e-3) on both,
and the pictures were byte-identical between graph and eager output. The same-job re-run on the merged
head (after main's card-89 merge) is `bench-122-ab.sh` at `2eaf18ad6` — `git apply -R` of
`bench-122-base.patch` puts
the card's h3 files back to main's inside the job's shallow clone: job `top-ae939c0b8de7` (exit 0) ran
the merged head's warm mean **15.3 s** (15.3, 15.4, 15.3: transformer 12.9, steps 2.13-2.15, decoder
1.3) against the same-job `main + --tc` baseline's **15.5 s** (15.4, 15.8, 15.4: transformer 12.9,
steps 2.14-2.16) — 0.2 s faster, kept,
with `latent.f32` and the PNG byte-identical between phases and every selftest passing. Seconds
outside the steps are ~0.0 s on the graph path — the rows in/out work is captured in each step's
graph and the transformer total and the sum of the steps
agree to the printed hundredths (12.79 against 12.80); `H3_PROFILE`'s per-stage waits can't be
captured, so the graphed run has no finer breakdown than that.

Card 89's own tile, `h3_gemm_int8`, was measured and reverted. `h3_quantize` scaled each 128-wide chunk by its own
absmax, so `out` and `fc2` left bf16 without the per-row scale's distance from episode 1, and the `m16n8k32` tile
folded a chunk's exact int32 sum into the fp32 accumulator at that chunk's scales, writing bf16 from its own tile —
no int32 round trip through `h3_dequantize`, `fc1`'s product feeding a bf16 `h3_swiglu`. All four projections ran it
behind `--int8`. `cdf02b8b4` (job `top-07d5c772c895`, `bench-int8.sh`, 19 November, exit 0): every selftest passed,
the decoder's same-latent `reference` repeated `96c87a43`'s bound (86.031% of bytes identical, 99.908% within 1, at
most 5 apart, PSNR 56.59 dB, relative error 3.20e-3 against 3e-2), and the picture passed every gate — latent cosine
0.9959, 29.51 dB and 3.39 apart on average from episode 1's Mac PNG, six steps. But a warm 2048² picture took
**20.5 s** mean (20.4, 20.4, 20.6: prompt 1.1, transformer 14.4, decoder 4.9), the projections running q k v gate
3.04 s, attention out 1.11 s, mlp in 4.52 s, mlp out 2.34 s against `c30403d9`'s 18.9 s default (job
`top-85cea73cbb20`, transformer 12.8: 2.68, 1.14, 3.38, 2.27). The own tile is slower on q k v gate (+0.36), mlp
in (+1.14) and mlp out (+0.07) and a hair faster on attention out (−0.03); the 20.5 s against 18.9 s is the
revert's ground, and the cuBLASLt qkvc/fc1 int8 default stands. `bench-int8.sh` named the setting through the
Desk's env-less door at `cdf02b8b4`; the revert removed it with the flag it set. The `bf16` window that job ran
beside (job `top-c5dfb77a467c`, at `b4c709177`, whose default was bf16) took 21.0 s (transformer 14.7).

The loss was named on TOP (`profile-int8-job.sh` at `38422b4b`, job `top-6384892e821f`; the job image's `ncu` is a
shim without its nsight-compute support dir, so the dynamic readout is the fatbin's `cuobjdump -res-usage` and the
measured per-stage times above, not a profile). The reverted tile staged both scale vectors in shared and folded each
chunk's int32 sum into fp32 at about five ops a element; it ran **167 registers** at 256 threads and 3 × (16,384 +
16,384 + 1,024) = **101,376 shared bytes a block** — sm_121's per-block opt-in cap (`static_assert(SMEM <= 101376)`
in `deepseek_v41/attn_cuda.cu`), and its ~100 KB of shared an SM caps residency at one 128×128 block, eight warps
(16.7% of the 48-warp scheduler; 167 registers × 256 threads says one block on its own) to hide each chunk's
ldmatrix and mma latency; cuBLASLt's heuristic choice pipelines deeper and holds ~154 TOPS where this tile ran
slower (its share was inferred from the 20.5 s warm wall, ncu being unavailable).
The fix keeps the m16n8k32 tile but folds only the activation scale in registers (two fma a element against the
12582912.0f magic bias), applies the weight scale once in the epilogue and stages only that vector — the rebuilt
`h3_gemm_int8` below.

Card 129 rebuilt the tile on that diagnosis — one activation scale folded in registers, the weight scale once in the
epilogue — and its contract gates all pass (`bench-129-contract.sh` at `fd2ecf13e`, job `top-56c1a215a30f`): but on
the picture's own shapes it still loses to cuBLASLt, so it stops short of a picture run by this card's own rule.
`h3-int8-bench` on TOP (same job): qkvc 116.0 vs 152.1 TOPS, out 119.2 vs 151.3, fc1 115.1 vs 153.5, fc2 118.5 vs
149.6 — a steady ~0.75-0.79× of cuBLASLt's int8 rate. The occupancy fix was real (one scale vector staged instead of
two, ~2 fma an element instead of ~5) and the tile is correct, but a third fewer TOPS than the library it would
replace is not a win. The tile went back out of the tree; the win that remained looked like coverage, not rate —
`out` and `fc2` joined `qkvc` and `fc1` on the same cuBLASLt int8 path — but the picture gates said no below.

Two more tile geometries were run under the same stop rule (`bench-129-contract.sh` at `3e1c5842f` job
`top-3abb0a0b31cb` 82.6/81.6/83.8/78.3 TOPS for a 256×128 two-stage tile with a per-step n-pair fold — worse
than 128×128, the wider tile's extra smem stage lost to its fold running four times a chunk; at `8c4f3cb89` job
`top-5688a4933512` a 192×128 two-stage tile at 12 warps an SM measured 116.0/118.1/117.6/115.9 TOPS, 0.74-0.77×
cuBLASLt — it recovered to the 128×128 plateau, no better). Three geometries now bracket the same ~117 TOPS
ceiling: a cp.async tile with eight-to-twelve warps and no persistent grid tops out a quarter under cuBLASLt's
~152, and mmh3's own 172 comes from TMA plus a persistent grid, not the tile shape — that port is its own card.

The rescope won on speed but not on the picture: `out` and `fc2` joined `qkvc` and `fc1` on the cuBLASLt int8
path (`bench-129-cublaslt.sh` at `6279e2a2`, job `top-ee9f603eeeed`, 20 November, exit 0, `H3_PROFILE=1` on
both sides — no CUDA graph, so the warm walls are ~1 s above the same commits graphed). All four projections int8
gave a warm 2048² **13.7 s** (13.7, 13.6, 13.7; transformer 12.3) against the same job's main re-measure
**14.3 s** (14.3, 14.3, 14.3; transformer 12.9): the 286 TFLOP of `out`/`fc2` products moved off bf16 saved
0.6 s of kernel time. Every selftest passed on both sides; the same-latent `reference` passed on the after
side (85.246% bytes identical, 99.887% within 1, at most 5 apart, PSNR 56.33 dB, relative error 3.31e-3
against 3e-2). But `h3_image_compare.py` on the fetched picture measured **latent cosine 0.9945, PSNR 28.15
dB, mean diff 4.23** against episode 1's Mac latent and PNG — cosine over 0.99, the other two under the gates
(28.2 dB, 4.1), the same 28.15/4.23 `bd734252` measured the day before when every projection ran int8. Per-row
scales are the only fold cuBLASLt's int32 C supports; the finer scales the gates need there want the own-mma
product this card stopped. `out` and `fc2` went back to bf16, so the tree carries qkvc+fc1 int8 and the card
closes at main's warm time with the measured receipts above.

The rounds left toward mmh3's 9.5 s warm time (≤ 7 s by card 93; its first picture is 17.6 s, card 80's standing
goal), in order, each needing its own window:

1. **The int8 products' own rate** is the 1.3 s left: `qkvc` and `fc1` take 2.68 and 3.39 s a picture at ~94 TFLOPS of
   int8, a third of the GB10's dense rate, and their int32 sums make a round trip through `h3_dequantize`. A product
   that writes bf16 straight from the tile with a per-128-chunk scale was card 89's `h3_gemm_int8`: it measured 20.5 s
   warm against `c30403d9`'s 18.9 s (above) and is reverted, so the cuBLASLt products stay. A faster int8 product than
   cuBLASLt's heuristic first offer is still the round's open part — `822b217c` found the heuristic's first offer
   already its fastest, and card 129 found three cp.async geometries plateau around 117 TOPS (0.74–0.79× cuBLASLt):
   matching mmh3's ~172 needs TMA plus a persistent grid, a port of its own, not a tile-shape retry. The two kernels'
   records are in the registry as of `cadab6b2`, and `H3_BUILD=contract sh docs/recipes/minimax-h3-cuda/cuda-build.sh`
   runs their gates 1-4 (`contract_h3_int8.zig`); the limits audit (gate 5) is `limits-job.sh`'s own window.
2. **Attention**: 1.80 s a picture in the transformer on main (`368cee6b`'s re-pipelining measured 1.91 s, reverted
   `787f3d3d`), 217 registers at 128 threads and two CTAs an SM (8 warps), and 1.89 s in the decoder, whose
   16-byte key loads changed nothing. Its bound is not the value load's latency — deeper prefetch and query-in-registers
   both lost to register pressure — so mmh3's INT8/FP8 attention is the candidate that would also cut registers (the
   `fb5090ae` fp8 round measured 14.1 s a picture and is reverted); the
   decoder's attention needs its stall found first, since neither its loads nor its occupancy explain it.
3. **The rows outside the blocks**: on the device as of `bd734252`, 0.15 s a picture.
4. **The decoder's products** are 2.26 s at 121 registers and two CTAs an SM. A 512-thread 4 x 8 tile measured 3.00 s
   (`cb217aa2`, reverted): the smaller per-thread tile reads 12 shared floats for 32 products where 8 x 8 reads 16 for
   64, so the shared-memory rate rises faster than the warps help. A tile that keeps 8 x 8's ratio at more warps would
   need more registers than a 1024-thread SM allows. Tensor cores stay out: the selftest's 90% block identity
   (86.952% with them, card 72) fails.
5. **The 4096² memory round**: the decoder's buffers do not fit beside the resident prompt (round 2's OOM).

### Card 139: `out` and `fc2` int8 at a scale per 256 of K, through cuBLASLt batch — measured and reverted

Card 136's picture gates held on the Mac at a scale per 256 values of K on both operands where card 129's per-row
scales failed here (28.15 dB, mean 4.23 against 28.2 dB and 4.1). cuBLASLt's int8 sums are int32 only, so the fine
fold reached the library as a strided batch: `out` (K=7,168 → 28 groups) and `fc2` (K=14,336 → 56) ran one batched
call whose batch `g` summed the 256-deep slice `[g·256, g·256+256)` of K into its own (rows, n) int32 plane
(`Spec.batch`/`sa`/`sb`/`sc`), and `h3_dequantize_grouped` folded the planes at both operands' `S[row·G+g]` scales.
`h3_quantize` took the group count (1 the row's scale; K/256 the grouped one), staging the 128-chunk maxes in shared.

`bench-139.sh` at `6bf561216` (job `top-1c32bccda27d`, 20 November, exit 0, `H3_PROFILE=1` on both sides — no CUDA
graph) ran the bench on that commit then on `bench-139-base.patch` reverse-applied (main's qkvc/fc1-int8 default).
Every selftest passed on both sides. Grouped all-four int8 gave a warm 2048² **34.8 s** (34.8, 34.8, 35.0;
transformer 33.6: attention out 7.94, mlp out 16.12) against the same job's main re-measure **14.3 s**
(14.3, 14.3, 14.3; transformer 13.0: attention out 1.15, mlp out 2.35). The planes are the cost: `out` writes and
reads 28 (4,120 × 5,376) int32 planes and `fc2` 56 — ≈4 TiB of acc traffic a picture at 2048², an order over the
compute the int8 saves; a library fold needs the scales inside the product, which only an own-mma tile can do.
`h3_image_compare.py` on the fetched picture measured **latent cosine 0.9935, PSNR 28.02 dB, mean diff 4.27**
against episode 1's Mac latent and PNG — the grouped scales did not reach the picture either (main's bf16 `out`/`fc2`
side: cosine 0.9952, 28.69 dB, 4.01). The forward path is reverted; `out` and `fc2` stay bf16. Per-256-of-K scales
through cuBLASLt are closed: the library's int32 C writes each group's partial to its own plane, so a grouped fold
costs G n-sized round trips — only an own-mma product folding scales in registers can hold the gates and the rate.
Card 129's three tile geometries plateaued at 0.74–0.79× of cuBLASLt's int8 rate, so the round's open part stays a
TMA + persistent-grid int8 product at mmh3's ~172 TOPS, not a library call.

### The side-by-side: mmh3's warm server beside ours, unseen prompts (20 November)

[`side-by-side-job.sh`](minimax-h3-cuda/side-by-side-job.sh) puts both engines' resident servers
in one TOP window (job `top-d2fb1f45a475` at `0e2a3a5b`, GB10, driver 580.173.02) on the same five
prompts in the same order: the pilot's fox (episode 1's gate prompt), the three files in
[`prompts/`](minimax-h3-cuda/prompts) neither server has seen — a harbour, a meadow, a
locomotive — then the fox once more. Seed 42 and the checkpoint's six distilled steps throughout;
mmh3 runs its defaults (the six sigmas, dense BF16 attention, INT8 ConvRot linears, fp16 VAE) and
ours the default build (qkvc and fc1 int8 on cuBLASLt, the tensor-core decoder). Warm runs count
only what each engine pays for a prompt it has not seen; each side's fifth run repeats its own
first prompt, the repeated-prompt figure reported separately below and never part of the
headline. `bench-script.sh` learned `H3_BENCH_PROMPTS`, one prompt file a run (default: the pilot
prompt four times, as before).

| Run | Prompt | TensorFold wall | mmh3 wall |
|---|---|---|---:|---:|
| cold | fox | 153.3 s (load 51.3) | 17.6 s (17.2 server-reported) |
| warm 1 | harbour (unseen) | 15.1 s | 10.6 s (10.11 server) |
| warm 2 | meadow (unseen) | 15.2 s | 10.6 s (10.13 server) |
| warm 3 | steam (unseen) | 15.1 s | 10.1 s (9.62 server) |
| repeated | fox again, labelled cached | 13.9 s | 10.1 s (9.62 server) |

Warm walls are client-side; mmh3's server-reported `seconds` bracket the same request.
Per-run stage columns, warm:

| Stage | TensorFold unseen ×3 | mmh3 unseen ×3 |
|---|---:|---:|
| Prompt | 1.23-1.25 s | 0.2 s encode |
| Transformer, 6 steps | 12.5-12.7 s (~2.09 a step) | ~8.6-8.7 s (1.4-1.5 a step) |
| Decoder | 1.2-1.3 s | 0.7 s |
| Wall | **15.0-15.1 s** | **9.6-10.6 s** |

**mmh3 wins the window I ran.** The card's goal — a warm 2048² picture at or under 7 s beside
mmh3's 9.5 s — is not met at this head: I measure our warm at 15.0-15.1 s on unseen prompts (13.9 s on
the repeated one). mmh3's warm barely moves off its 9.5 s quoted figure when the prompt is new:
its INT8 text encoder stays resident and encodes an unseen prompt in 0.2 s (against its 3.4 s
first encode with the encoder load), so "unseen" costs it 0.2-0.5 s, not the 3.4 s card 125's
whole-run figure suggested. Our unseen prompt costs the BF16 encoder's 1.2 s; the kept-states hit
is 0.01 s. The transformer holds the gap that matters — its 8.6 s to our 12.6 s, the products and
the fused elementwise passes the rounds above already name — the decoder its 0.7 s to our 1.3 s.

The gates hold for the gate prompt: I measured run 1's latent cosine against episode 1's Mac
latent at **0.9952** (≥ 0.99) and its PNG **28.69 dB** PSNR and **4.02** apart on average (≥ 28.2,
≤ 4.1); run 5's is byte-identical. Every selftest passed and the same-latent `reference` compare
repeated the `--tc` bound: 85.213% of bytes identical, 99.875% within 1, at most 7 apart, 56.31
dB, relative error 3.33e-3 under 3e-2. Six steps both engines, same seed; mmh3's seed draws other
noise, so its side is times, not pixels.

What each engine does that the other doesn't, at this head:

- **TensorFold**: every denoising step replays as one captured CUDA graph — rows in, all 50
  blocks' launches, the step's final norm, output head and Euler — while mmh3 launches every
  kernel itself. The decoder's products and attention run on the tensor cores as cuBLASLt bf16
  GEMMs with fp32 sums (bias and LayerScale folded into GEMM epilogues and `h3d_res_norm`), all
  121 tiles of a 2048² canvas in one batched pass; mmh3's VAE is CUTLASS fp16. The prompt server
  keeps a prompt's states keyed by its exact bytes (LRU inside `H3_PROMPT_CACHE`); mmh3 keys on
  the encoder's digest.
- **mmh3**: all four linears run ConvRot INT8 `m16n8k32` mma products writing bf16 from the tile
  at ~172 TOPS with no int32 round trip — ours runs `qkvc`/`fc1` on cuBLASLt int8 (~154 TOPS, the
  int32 sums round-tripping `h3_dequantize`) and `out`/`fc2` still in bf16. Its one
  `add_norm_quantize` launch family is 1.34 s a picture to our fused set's ~2.4 s. Its text
  encoder is INT8 too and stays resident, so an unseen prompt costs 0.2 s where ours costs 1.2 s.
  It saves cuBLASLt's picks and its INT8 tilings to disk (`~/.cache/mmh3`), so a cold request pays
  no algorithm search; our cold picture pays one 4-7 s search on its first step (card 141 owns
  the same cache for us).
- **Neither side runs the other's headline trick**: where the card names block-scaled FP8/NVFP4
  projections and fused quantized attention, both engines were measured there and neither ships
  it on this model — our MXFP8 products missed every gate (`160c7687`, cosine 0.9664, 11.5 apart
  mean) and the e4m3/INT8 attention rounds measured slower and failed the picture (`acda1e04`,
  `fb5090ae`), while mmh3's own FP8/NVFP4 paths are opt-ins that change the picture, off in its
  defaults. On this checkpoint the gates, not the formats, decide.

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
ways, and what each of its tricks would save our warm 15.5 s picture. [`mmh3-server-job.sh`](minimax-h3-cuda/mmh3-server-job.sh)
runs `mmh3 server` on TOP's GB10 with the `cuda,server` feature: one cold picture, three warm repeats of the same
prompt (its states kept, 0 s) and one 1024² request. The second server under the CUPTI preload (`cupti-trace.c`)
traces every kernel launch for the warm segment's top-kernel table — nsys is absent from the job image.
[`bench-trace-job.sh`](minimax-h3-cuda/bench-trace-job.sh) runs our own `bench-script.sh` under the same preload
in the same kind of window, so both engines' kernel budgets sit in one table in the study.

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

`5d4bd40` (job `top-e611c4df9602`, 20 November, exit 0 — a same-job A/B: `bench-script.sh` on this commit,
then `bench-128-base.patch` reverse-applied and `bench-script.sh` again; the branch carries cards 124, 125, 127
and 131 merged from main) fuses the int8 transformer's elementwise passes as `docs/mmh3-h3-image-study.md`'s
lever 2 reads mmh3: `h3_norm_quant` runs the gated add, modulated RMSNorm, rotate and quantize as one launch
(the normed row is recomputed per lane as `modulated` rounded to bf16 — never materialized), and `h3_heads_i8`
reads the qkvc product's int32 sums and scales and runs `h3_heads`' q/k norm and rotary on values dequantized
as `h3_dequantize` rounds them — once a row, not per tile pair. `h3_swiglu_int8` stays its own launch over
fc1's sums: mmh3's swiglu sits in its own int8 GEMM's epilogue, and cuBLASLt's i8×i8→i32 descriptors take none
(the card's "in fc1's epilogue" clause asks for the amendment that keeps it a launch). `fused_test` in
`--self-test` checks both against the unfused chain byte for byte (transformer PASS in both phases), and all
eight bench latents are sha256-identical (`8d067d75…`) between head and baseline. A warm 2048² picture took
**13.9 s** mean (13.9, 13.9, 14.0: transformer 12.5–12.6, decoder 1.3–1.4 s) against the same job's baseline
at main **14.3 s** (14.3, 14.3, 14.3: transformer 12.9): `q k v gate` 2.68→1.88–1.89 s, `heads`
0.96→1.40–1.41 s, `mlp in` 3.39→3.32–3.34 s — the study's 3.20 s of elementwise launches is 2.4 s. Against
episode 1's Mac picture the gates hold unchanged — the latent is byte-identical to the measured one.

`b163d44` (job `top-52a99a9e86fd`, 20 November, exit 0 — a same-job A/B: `bench-145-ab.sh` on this commit, then
`bench-145-base.patch` reverse-applied and the same unseen prompts again) cuts the two non-product passes card
128's split left: `heads` 1.40 s and the ~0.8 s `h3_swiglu_int8` inside `mlp in`. `h3_heads_i8` keeps q and k
rounded to bf16 two a word (bit-identical values, half card 128's live registers — its float arrays took 252
and spilled), reads every row 16 bytes at a time, and reloads each rotary partner's `int32` group to dequantize
it again rather than sharing rounded values; `h3_swiglu_int8` runs one block a row of 4-wide `int4`/`float4`
groups instead of a grid-stride element loop, the same dequantize, silu and product. An int8 act rows variant
(swiglu quantizing its own output, fc2's product reading the codes) was cut before the run: cuBLASLt's mixed
bf16×i8 path has no epilogue for the row scale, so it could not keep the product bit-identical. `fused_test`
now checks `h3_swiglu_int8` against `h3_dequantize`+`h3_swiglu` too (transformer PASS in both phases). Each
phase renders an uncounted warm-up prompt, then three unseen 2048² prompts; the three latents are
byte-identical between phases (`e95fd65d…`, `58db502a…`, `a8beb316…`). A warm picture took **14.49, 14.53,
14.55 s** (transformer 11.98, 12.02, 12.03) against the same job's main at **15.02, 15.15, 15.13 s**
(transformer 12.52, 12.64, 12.63): `heads` 1.40–1.41→1.11–1.12 s, `mlp in` 3.28–3.30→3.23–3.25 s and `mlp out`
2.43–2.45→2.22–2.23 s, with `q k v gate` (1.84–1.85 s against 1.83–1.85), `attention` and `norms` unchanged.
`mlp out` is only fc2's bf16 product over `act`, so its 0.2 s comes from swiglu's new write order, likely
leaving `act` in L2 for fc2 (inferred, not profiled). Together that is ~0.55 s off card 128's ~2.4 s of
non-product passes, so ~1.85 s against mmh3's 1.34 s.
