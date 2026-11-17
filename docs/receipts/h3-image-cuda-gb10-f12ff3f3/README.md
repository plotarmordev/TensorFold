# MiniMax-H3-Image's CUDA selftests on a GB10

`tf-h3-image-decoder-cuda selftest` and `tf-h3-image-prompt-cuda selftest`, built at `d329fd2e` with
`-Dsm=121 -Doptimize=ReleaseFast`, on an NVIDIA GB10 (sm_121, driver 580.173.02). The run's checkout was
`f12ff3f3` (that commit plus the job script); `6740eb813` adds only this receipt and the recipe, so the
tool code is the same.

- `decoder-selftest.log`: 37 checks pass. Every block of the 24 is gated at 90% identical to the float64
  reference on the GPU's own input; the lowest is block 9 at 92.037%, worst 2.00 ULP, relative 8.87e-4.
  The pictures pass the Metal decoder's bytes bound (80% identical, at most 2 apart): 88.889% and
  92.164% identical, at most 1 apart. 25 tiles of 16 x 16 take 43.4 ms a tile.
- `prompt-selftest.log`: every gate passes; the worst row is 3.16e-2 of its row's RMS on
  `condition_proj`, the refiner and the final norm.

Both tools build synthetic models and compare against the host's float64 reference; no weights are read.
