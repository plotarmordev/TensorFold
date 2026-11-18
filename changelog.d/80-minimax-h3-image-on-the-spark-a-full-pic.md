### MiniMax-H3-Image on CUDA: AdaLN tables kept across served pictures

- `tf-h3-image-cuda --serve` keeps the AdaLN tables between pictures with the same timesteps instead of re-reading ~26 GB of projection weights for each one; card 68's warm 2048² picture spent 26.6 s of its 77.0 s there.
- The CUDA recipe lists the remaining rounds toward mmh3's 17.6 s on the same Spark: a resident prompt stage, a resident decoder, then INT8 linear layers behind the Mac-render gates.

### MiniMax-H3-Image on CUDA: the prompt stage's weights stay on the device

- `tf-h3-image-prompt-cuda` uploads the text model's layers, the refiner's front end and its blocks into one device arena once (`prompt_cuda.loadResident`), evicts the shards' pages, and serves pictures with `--serve`; a warm picture's prompt no longer copies ~45 GiB through the pinned staging a layer at a time.
- `bench-script.sh` keeps that server beside the transformer's, so the first picture of a size pays the upload and the three warm repeats do not.
- The CUDA recipe records Desk's `fd812222` window: a warm 2048² picture at 49.7 s (prompt 16.3, transformer 24.4, decoder 1.7 + 5.8) against mmh3's 17.6 s, and the rounds left in order.

### MiniMax-H3-Image on CUDA: the decoder resident across served pictures

- `tf-h3-image-decoder-cuda --serve` loads the VAE's weights once and decodes one latent line per picture, so a warm picture pays its tiles and PNG but not the 1.7 s load; `bench-script.sh` keeps that server beside the prompt's and the transformer's for each size.

### MiniMax-H3-Image on CUDA: the prompt's products read each weight once, coalesced

- `h3p_linear` now gives each output column a warp whose lanes load the column's k weights in coalesced
  32-element windows into shared memory, and each lane sums one row over them in the reference's k order. The
  old thread-per-output kernel read every weight row once per output row from a 10 KB stride, so the pilot
  prompt's 24 rows re-read the text encoder's ~45 GiB of weights 24 times, uncoalesced: the resident prompt
  cost 35.8 s warm at 2048² and 9.5 s at 4096² instead of the second the residency was meant to buy.
- The products' sums keep the sequential fp32 dot and every value's rounding, so the text rows are unchanged.

### MiniMax-H3-Image on CUDA: the prompt's window widened to 256 weights a warp

- `h3p_linear` streams each output column through a 256-element shared window, eight weights a lane as one
  16-byte load, instead of 32: 512 bytes stay in flight per warp and eight products follow each load, while
  the dots keep the reference's sequential fp32 order so the text rows are unchanged. The 32-element window
  still left the resident prompt at 5.48 s (8 GB/s, memory-latency bound) in Desk's `9cf468df` window.
- `bench-script.sh` runs the decoder's `reference` compare after the resident servers close, so the host
  decoder gets the memory they held: the `9cf468df` job thrashed there and never reached 4096². It measures
  2048² by default (`H3_BENCH_SIZES='2048 4096'` runs both), since the 4096² decoder's buffers pass the
  GB10's memory beside the resident prompt (round 2's OOM) and wait on their own round.

### MiniMax-H3-Image on CUDA: the prompt at 1.1 s, a warm 2048² picture at 31.4 s

- The 256-weight window (`c4673100`) took the resident prompt from 5.48 s to **1.1 s** warm (45 GiB at 41 GB/s) in
  Desk's `f825b43c` window: a warm 2048² picture fell from 35.3 s to **31.0 s** (prompt 1.14, transformer 24.21,
  decoder 5.60); cold 178.3 s.
- `cecba1009` repeated it with the fixed script: every selftest passed, a warm 2048² picture took **31.4 s** mean
  (prompt 1.1, transformer 24.5, decoder 5.7), cold 170.3 s, peak 86.8 GiB, 64.9 W and 2.0 kJ a picture, and the job
  exited 0. Its `reference` compare passed with `96c87a43`'s numbers: 86.180% of bytes identical, 99.923% within 1,
  at most 5 apart, PSNR 56.65 dB, relative error 3.15e-3 against the 3e-2 bound.
- Asking cuBLASLt's heuristic at the picture's 4,120 rows instead of the canonical 16,384-row batch measured the
  same transformer (23.91–24.54 s against 24.13 s), so that round is reverted and the upstream arithmetic policy
  stays.
- `bench-script.sh` closes its FIFO fds in each resident server: the decoder server, started last, held the
  transformer's and prompt's request write ends, so closing them never gave those servers EOF and `wait` hung after
  the 2048 runs. `9cf468df`'s ~1.5 h hang was that leak, not the 4096² stage.
- INT8 linear layers are the one round left for mmh3's 17.6 s: at its 1.4 s a step the six steps take ~8.5 s and a
  warm 2048² picture sits near 15.2 s with today's prompt and decoder.

### MiniMax-H3-Image on CUDA: `--int8` projections, a warm 2048² picture at 28.9 s

- `tf-h3-image-cuda --int8` runs the blocks' projections as mmh3 does: weight and activation rows Hadamard-rotated in
  128-wide chunks, int8 at their absmax, exact int32 sums in cuBLASLt (`Gemm.runInt`), one bf16 rounding back; the bf16
  projections are freed once quantized. `bench-script.sh` runs it (`H3_LINEAR=bf16` for the old path).
- Desk's `df74446f` window: a warm 2048² picture at **28.9 s** (prompt 1.07, transformer 22.2, decoder 5.6) against 31.4 s,
  the latent at cosine 0.9961 against BF16's. A step fell only from 4.07 s to 3.69 s, so the next round profiles the
  step under `--int8` before attention; BF16 stays the default until the Mac-render gates pass on the `--int8` picture.

### MiniMax-H3-Image on CUDA: a step's host rows on every core, a warm 2048² picture at 20.6 s

- `H3_PROFILE` makes `tf-h3-image-cuda` print each picture's block stages and the time outside them; the bench sets it.
  The `--int8` step's 1.5 s outside the blocks was the host's serial `rows_in`, final norm and `rows_out`.
- `tf-h3-image-cuda` splits those rows over the host's cores, each row's sums in the serial order (the latent is
  byte-identical): the transformer fell from 21.8 s to 13.9 s and a warm 2048² picture to **20.6 s** (prompt 1.07,
  transformer 13.9, decoder 5.6), 3 s above mmh3's 17.6 s.

### MiniMax-H3-Image on CUDA: the decoder's products load 16 bytes and prefetch, a warm 2048² picture at 20.3 s

- `h3d_linear` loads eight K of a row as one 16-byte load and prefetches the next K slice into registers while the
  current one sums, in the same order: the decoder's GPU time fell from 5.25 s to 4.95 s with its selftest rows
  unchanged, and a warm 2048² picture to **20.3 s** (prompt 1.1, transformer 13.9, decoder 5.3) in job
  `top-9d030e00d7aa`, every selftest passing.
- `bench-script.sh` builds only the sm_121 tools (`H3_BUILD=sm121 cuda-build.sh`); the five release SMs and their
  limits audit stay `cuda-build.sh`'s, and a bench round now takes ~10 minutes against the Desk's 20.

### MiniMax-H3-Image on CUDA: the decoder's products fuse into one rounded fma

- `h3d_linear` sums each product as `__fmaf_rn(a, b, acc)`: one rounding a product instead of two, so the sums sit
  closer to the float64 reference the decoder selftest compares against, at half the instructions of the unfused
  multiply and add it replaced. The selftest's gate is unchanged (90% of a block's bf16 values identical, worst 8
  ULPs, relative 2e-3).
- `tf-h3-image-decoder-cuda` prints the decoder's per-stage seconds under `H3_PROFILE` (`embedding`, `norms`,
  `products`, `finish`, `heads`, `attention`, `swiglu`), with the same waits `tf-h3-image-cuda` makes, so the next
  decoder round starts from a measured split instead of a guess.
- `h3_quantize` and `h3_dequantize`, the two launches `tf-h3-image-cuda --int8` adds, are in the kernel registry with
  the usage nvcc printed. `limits-job.sh` (`cuda-build.sh` with `H3_BUILD=limits`) runs the audit as its own TOP job:
  the five release SMs and `kernel-contract-limits`, no picture tools, CLI or `tf-cuda-test`, so it fits the Desk's
  20 minutes and never sits inside a timed bench.
- Desk's `top-c3142e924abc` window at `b4c5c42` (exit 0): a warm 2048² picture at **19.9 s** mean (19.9, 19.9,
  20.0), cold 161.6 s, peak 71,147 MiB. The decoder's 4.54 s of GPU time split into products 2.26 s, attention
  1.88 s, finish 0.25 s, swiglu 0.08 s, heads 0.04 s, norms 0.02 s and embedding 0.00 s. Every selftest passed and
  `reference` passed at 86.044% of bytes identical, 99.917% within 1, at most 6 apart, PSNR 56.60 dB, relative
  error 3.16e-3 against the 3e-2 bound. The fma's decoder fell from 5.3 s to 4.9 s (4.95 to 4.54 on the GPU); the
  transformer stays at 13.9 s, so the gap to mmh3's 17.6 s is its int8 products, its attention and the host rows.

### MiniMax-H3-Image on CUDA: the int8 kernels' gates, the served tables' regression

- `zig/tests/cuda/contract_h3_int8.zig` puts `h3_quantize` and `h3_dequantize` through gates 1-4 against float64:
  the 128-wide rotation's Sylvester signs, the absmax scale, every code within a rounding tie, the stated infinity
  and NaN rules, five byte-identical repeats and eight shapes each. `check-job.sh` runs those gates and
  `serve-check.sh` (one resident transformer renders 512² cases a, a, b, a; every latent must equal its case's
  fresh run byte for byte) as one TOP job.
- Job `top-de7892d85798` at `1c5074d3` (exit 0): both kernels passed, `h3_dequantize`'s shapes within 0.996 of
  their bound, the live sm_121 limits passed, and served 1-4 matched fresh a, a, b, a. `serve-check.sh` writes
  case b (a plus a 2 x 2 reference grid and its anchors) in safetensors' stdlib format, since TOP's python3 has
  no numpy.
