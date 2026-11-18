# MiniMax-H3-Image's CUDA selftests on a GB10

`tf-h3-image-decoder-cuda selftest` and `tf-h3-image-prompt-cuda selftest`, built at `0bfee52e` with
`-Dsm=121 -Doptimize=ReleaseFast`, on an NVIDIA GB10 (sm_121, driver 580.173.02). The run's checkout was
`0b573dd2` (that commit plus the job script), so the tool code is the same.

- `decoder-selftest.txt`: every check passes. Every block of the 24 is gated at 90% identical to the float64
  reference on the GPU's own input; the lowest is block 9 at 92.037%, worst 2.00 ULP, relative 8.87e-4.
  The pictures pass the Metal decoder's bytes bound (80% identical, at most 2 apart): 88.889% and
  92.164% identical, at most 1 apart. 25 tiles of 16 x 16 take 43.6 ms a tile.
- `prompt-selftest.txt`: every gate passes; the worst row is 4.66e-2 of its row's RMS on
  `condition_proj`, the refiner and the final norm.

Both tools build synthetic models and compare against the host's float64 reference; no weights are read.

The same run's kernel contract:

```text
SUMMARY 685 kernels: 685 pass gates 1-4, 0 fail, 8 known findings; 422 controls: 422 red as planted, 0 missed; 188 limit problems; gates 3449547 ms, limits 765095 ms
```
