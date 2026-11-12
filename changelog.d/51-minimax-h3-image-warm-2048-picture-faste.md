# MiniMax-H3-Image: resident weights

- `tf-h3-image-server` holds every weight file it maps, and the decoder's weights, in one Metal residency set on the prompt, transformer and decoder queues, so macOS no longer unwires them between a request's stages. A warm 2048x2048 picture on an M5 Ultra takes 12.7-12.8 s, down from 13.2-13.3 s on the same machine and load, and 13.8-16.6 s on busier runs. The latent and PNG are byte-identical. Weights past 70% of RAM leave the server running as before; a request after an idle pause still re-wires once, in its prompt.
