# MiniMax-H3-Image on CUDA: the decoder under mmh3's, every product fusing its bias

- A CUPTI trace of main's warm 2048² decode (job `top-bec0983c41cc`) put 0.754 s in kernels: products 347 ms,
  `h3d_finish` 236 ms, SwiGLU 80, heads 35, norms 30, `h3d_flash` 24. The VAE decoder is a ViT: no convolutions.
- `h3d_finish` ran because cuBLASLt's first pick for three of the four product shapes refuses the bias epilogue.
  `Linear.repick` asks the heuristic again with the epilogue set, and the decoder's products (`Gemms.planFused`) take
  an algorithm that fuses it; other engines' picks are unchanged. Two banded passes are now in flight, the host
  placing one's tiles from pinned rows while the GPU runs the next.
- Same-job A/B (`bench-193-ab.sh`, job `top-dba6bd470f74`): the warm 2048² decoder stage is 0.59 s against main's
  0.88 s (−33%, GPU 0.52 s against 0.76 s; mmh3's is ~0.7 s), cold 1.01 s against 1.22 s.
- Every latent and PNG (the fox and three unseen prompts) is byte-identical to main's, so the fox keeps cosine 0.9952,
  28.69 dB and 4.015 against episode 1's Mac picture. The same-latent `reference` held at 56.43 dB, relative error
  3.26e-3 under 3e-2, and the `h3d_flash` contract gates passed.
