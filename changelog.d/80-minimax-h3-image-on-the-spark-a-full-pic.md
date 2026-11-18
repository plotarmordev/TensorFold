### MiniMax-H3-Image on CUDA: AdaLN tables kept across served pictures

- `tf-h3-image-cuda --serve` keeps the AdaLN tables between pictures with the same timesteps instead of re-reading ~26 GB of projection weights for each one; card 68's warm 2048² picture spent 26.6 s of its 77.0 s there.
- The CUDA recipe lists the remaining rounds toward mmh3's 17.6 s on the same Spark: a resident prompt stage, a resident decoder, then INT8 linear layers behind the Mac-render gates.

### MiniMax-H3-Image on CUDA: the prompt stage's weights stay on the device

- `tf-h3-image-prompt-cuda` uploads the text model's layers, the refiner's front end and its blocks into one device arena once (`prompt_cuda.loadResident`), evicts the shards' pages, and serves pictures with `--serve`; a warm picture's prompt no longer copies ~45 GiB through the pinned staging a layer at a time.
- `bench-script.sh` keeps that server beside the transformer's, so the first picture of a size pays the upload and the three warm repeats do not.
- The CUDA recipe records Desk's `fd812222` window: a warm 2048² picture at 49.7 s (prompt 16.3, transformer 24.4, decoder 1.7 + 5.8) against mmh3's 17.6 s, and the rounds left in order.

### MiniMax-H3-Image on CUDA: the decoder resident across served pictures

- `tf-h3-image-decoder-cuda --serve` loads the VAE's weights once and decodes one latent line per picture, so a warm picture pays its tiles and PNG but not the 1.7 s load; `bench-script.sh` keeps that server beside the prompt's and the transformer's for each size.
