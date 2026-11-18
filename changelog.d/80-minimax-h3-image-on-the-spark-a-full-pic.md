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
