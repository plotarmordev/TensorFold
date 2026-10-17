# Hold summary (`summarize.py`)

## Decode (second DGX Spark), in run order

| run | commit | server sha256 | fresh window costs | ready s | heat waits s | admission |
|---|---|---|---|---|---|---|
| base-d | 50a34190b04055ff9268cc12f9d8284323830a25 | 38cb28f3e07de241d172d7579c9b1b55a291bdfc45475506c2e5a7c443236dc4 | True | 34.0 | 10.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.97 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 32.5 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-d | a79678c5624e55f5b009700e4d953e8034de9336 | 692e2fd510491ae264fac0deaa66b6453eacc367adb50badeda332a3d1279d17 | True | 34.0 | 28.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.49 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 33.0 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-d2 | a79678c5624e55f5b009700e4d953e8034de9336 | 692e2fd510491ae264fac0deaa66b6453eacc367adb50badeda332a3d1279d17 | False | 18.0 | 18.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.49 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 33.1 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-d2 | 50a34190b04055ff9268cc12f9d8284323830a25 | 38cb28f3e07de241d172d7579c9b1b55a291bdfc45475506c2e5a7c443236dc4 | False | 16.0 | 42.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.97 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 32.6 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |

| run | code / chat token_sha256 (drafted == draft:false) | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|---|
| base-d | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 47.14 (47.13, 47.14, 47.15); gpu-chat-no-think 36.40 (36.4, 36.4, 36.39); aggregate 47.1, 137.0, 36.4, 99.4 | 30 (0) / 4 (0) |
| pr-d | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 47.02 (47.01, 47.03, 47.02); gpu-chat-no-think 36.21 (36.22, 36.2, 36.21); aggregate 47.0, 136.5, 36.2, 99.0 | 30 (0) / 4 (0) |
| pr-d2 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 47.14 (47.13, 47.16, 47.14); gpu-chat-no-think 36.32 (36.32, 36.31, 36.32); aggregate 47.2, 136.4, 36.3, 98.9 | 30 (0) / 4 (0) |
| base-d2 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 47.07 (47.07, 47.06, 47.09); gpu-chat-no-think 36.35 (36.36, 36.35, 36.35); aggregate 47.1, 136.7, 36.3, 99.1 | 30 (0) / 4 (0) |


## Cold prefill (first DGX Spark), in run order

| run | commit | server sha256 | fresh window costs | ready s | heat waits s | admission |
|---|---|---|---|---|---|---|
| base-1 | 50a34190b04055ff9268cc12f9d8284323830a25 | 38cb28f3e07de241d172d7579c9b1b55a291bdfc45475506c2e5a7c443236dc4 | True | 38.0 | 266.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 72.01 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 30.3 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-1 | a79678c5624e55f5b009700e4d953e8034de9336 | 692e2fd510491ae264fac0deaa66b6453eacc367adb50badeda332a3d1279d17 | True | 38.0 | 170.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.54 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 30.9 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-2 | a79678c5624e55f5b009700e4d953e8034de9336 | 692e2fd510491ae264fac0deaa66b6453eacc367adb50badeda332a3d1279d17 | False | 20.0 | 202.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.54 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 30.8 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-2 | 50a34190b04055ff9268cc12f9d8284323830a25 | 38cb28f3e07de241d172d7579c9b1b55a291bdfc45475506c2e5a7c443236dc4 | False | 20.0 | 304.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 72.01 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 30.3 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |

| run | prompt tokens | cold ttft_s: measured - logged heat wait = net | cached tokens | drafted token_sha256 (== draft:false) | qualified |
|---|---|---|---|---|---|
| base-1 | 2048 | 1.225 - 0.0 = 1.225, 1.213 - 0.0 = 1.213, 1.213 - 0.0 = 1.213 | [0, 0, 0] | 027891eacb707268 (True) | True |
| base-1 | 8192 | 4.439 - 0.0 = 4.439, 8.421 - 4.0 = 4.421, 8.422 - 4.0 = 4.422 | [0, 0, 0] | 80b2b503a848b4d0 (True) | True |
| base-1 | 32768 | 31.084 - 18.0 = 13.084, 33.059 - 20.0 = 13.059, 33.070 - 20.0 = 13.070 | [0, 0, 0] | 70fb5c3caf050d1c (True) | True |
| base-1 | 65536 | 67.877 - 42.0 = 25.877, 69.850 - 44.0 = 25.850, 71.833 - 46.0 = 25.833 | [0, 0, 0] | 2e8f1a6db1bf3a71 (True) | True |
| pr-1 | 2048 | 1.035 - 0.0 = 1.035, 1.032 - 0.0 = 1.032, 1.024 - 0.0 = 1.024 | [0, 0, 0] | 027891eacb707268 (True) | True |
| pr-1 | 8192 | 3.278 - 0.0 = 3.278, 3.267 - 0.0 = 3.267, 3.275 - 0.0 = 3.275 | [0, 0, 0] | 80b2b503a848b4d0 (True) | True |
| pr-1 | 32768 | 21.584 - 12.0 = 9.584, 23.494 - 14.0 = 9.494, 21.595 - 12.0 = 9.595 | [0, 0, 0] | 70fb5c3caf050d1c (True) | True |
| pr-1 | 65536 | 47.215 - 28.0 = 19.215, 49.204 - 30.0 = 19.204, 51.019 - 32.0 = 19.019 | [0, 0, 0] | 2e8f1a6db1bf3a71 (True) | True |
| pr-2 | 2048 | 1.042 - 0.0 = 1.042, 1.026 - 0.0 = 1.026, 1.028 - 0.0 = 1.028 | [0, 0, 0] | 027891eacb707268 (True) | True |
| pr-2 | 8192 | 3.290 - 0.0 = 3.290, 3.278 - 0.0 = 3.278, 3.285 - 0.0 = 3.285 | [0, 0, 0] | 80b2b503a848b4d0 (True) | True |
| pr-2 | 32768 | 21.647 - 12.0 = 9.647, 25.521 - 16.0 = 9.521, 23.617 - 14.0 = 9.617 | [0, 0, 0] | 70fb5c3caf050d1c (True) | True |
| pr-2 | 65536 | 51.272 - 32.0 = 19.272, 57.370 - 38.0 = 19.370, 53.297 - 34.0 = 19.297 | [0, 0, 0] | 2e8f1a6db1bf3a71 (True) | True |
| base-2 | 2048 | 1.230 - 0.0 = 1.230, 1.219 - 0.0 = 1.219, 5.210 - 4.0 = 1.210 | [0, 0, 0] | 027891eacb707268 (True) | True |
| base-2 | 8192 | 8.446 - 4.0 = 4.446, 8.435 - 4.0 = 4.435, 10.435 - 6.0 = 4.435 | [0, 0, 0] | 80b2b503a848b4d0 (True) | True |
| base-2 | 32768 | 35.113 - 22.0 = 13.113, 35.097 - 22.0 = 13.097, 35.092 - 22.0 = 13.092 | [0, 0, 0] | 70fb5c3caf050d1c (True) | True |
| base-2 | 65536 | 71.976 - 48.0 = 23.976, 69.899 - 44.0 = 25.899, 69.907 - 44.0 = 25.907 | [0, 0, 0] | 2e8f1a6db1bf3a71 (True) | True |

| prompt tokens | base net ttft_s, median (n) | PR net ttft_s, median (n) | PR / base | token_sha256 equal across runs |
|---|---|---|---|---|
| 2048 | 1.216 (6) | 1.030 (6) | 0.846 | True |
| 8192 | 4.435 (6) | 3.278 (6) | 0.739 | True |
| 32768 | 13.088 (6) | 9.589 (6) | 0.733 | True |
| 65536 | 25.864 (6) | 19.244 (6) | 0.744 | True |

## Cold prefill at 131,072 ids, in run order

| run | commit | server sha256 | fresh window costs | ready s | heat waits s | admission |
|---|---|---|---|---|---|---|
| base-L1 | 50a34190b04055ff9268cc12f9d8284323830a25 | 38cb28f3e07de241d172d7579c9b1b55a291bdfc45475506c2e5a7c443236dc4 | False | 20.0 | 360.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 72.01 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 30.2 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-L1 | a79678c5624e55f5b009700e4d953e8034de9336 | 692e2fd510491ae264fac0deaa66b6453eacc367adb50badeda332a3d1279d17 | False | 20.0 | 312.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.54 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 30.8 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-L2 | a79678c5624e55f5b009700e4d953e8034de9336 | 692e2fd510491ae264fac0deaa66b6453eacc367adb50badeda332a3d1279d17 | False | 20.0 | 336.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.54 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 30.6 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-L2 | 50a34190b04055ff9268cc12f9d8284323830a25 | 38cb28f3e07de241d172d7579c9b1b55a291bdfc45475506c2e5a7c443236dc4 | False | 20.0 | 362.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 72.01 GiB; 8 streams at once, 1.34 GiB each at a 131584-token window, of 31.1 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |

| run | prompt tokens | cold ttft_s: measured - logged heat wait = net | cached tokens | drafted token_sha256 (== draft:false) | qualified |
|---|---|---|---|---|---|
| base-L1 | 131072 | 135.111 - 78.0 = 57.111, 145.740 - 90.0 = 55.740, 151.777 - 96.0 = 55.777 | [0, 0, 0] | 29cb05e3461a3e2f (True) | True |
| pr-L1 | 131072 | 115.360 - 72.0 = 43.360, 122.386 - 80.0 = 42.386, 123.884 - 82.0 = 41.884 | [0, 0, 0] | 29cb05e3461a3e2f (True) | True |
| pr-L2 | 131072 | 124.055 - 82.0 = 42.055, 127.933 - 86.0 = 41.933, 127.926 - 86.0 = 41.926 | [0, 0, 0] | 29cb05e3461a3e2f (True) | True |
| base-L2 | 131072 | 139.491 - 84.0 = 55.491, 145.933 - 90.0 = 55.933, 149.967 - 94.0 = 55.967 | [0, 0, 0] | 29cb05e3461a3e2f (True) | True |

| prompt tokens | base net ttft_s, median (n) | PR net ttft_s, median (n) | PR / base | token_sha256 equal across runs |
|---|---|---|---|---|
| 131072 | 55.855 (6) | 41.994 (6) | 0.752 | True |
