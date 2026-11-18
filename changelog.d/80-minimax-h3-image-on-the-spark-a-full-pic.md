### MiniMax-H3-Image on CUDA: AdaLN tables kept across served pictures

- `tf-h3-image-cuda --serve` keeps the AdaLN tables between pictures with the same timesteps instead of re-reading ~26 GB of projection weights for each one; card 68's warm 2048² picture spent 26.6 s of its 77.0 s there.
- The CUDA recipe lists the remaining rounds toward mmh3's 17.6 s on the same Spark: a resident prompt stage, a resident decoder, then INT8 linear layers behind the Mac-render gates.
