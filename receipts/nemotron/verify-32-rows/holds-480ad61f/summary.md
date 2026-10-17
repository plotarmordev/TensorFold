| run | commit | server sha256 | code token_sha256 (drafted == draft:false) | chat token_sha256 (drafted == draft:false) |
|---|---|---|---|---|
| base-1 | 39993136981823d314a1a6204ecc6f95ae25bb45 | ef46e9ba4e9b05a763306f3c37ef99ec7bf77892ec657bb078ac37e3b49c6e7c | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-1 | 480ad61f5bb8507da6b051cf138b8e615c59a519 | 029382b0c0e504d31bb7bf64b1307de9ccba5e9d1a4c8c88e3aa9df6b2526564 | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-2 | 480ad61f5bb8507da6b051cf138b8e615c59a519 | 029382b0c0e504d31bb7bf64b1307de9ccba5e9d1a4c8c88e3aa9df6b2526564 | 51e4c630298a14a1 | 161034fd0a090a34 |
| base-2 | 39993136981823d314a1a6204ecc6f95ae25bb45 | ef46e9ba4e9b05a763306f3c37ef99ec7bf77892ec657bb078ac37e3b49c6e7c | 51e4c630298a14a1 | 161034fd0a090a34 |

| run | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|
| base-1 | fibonacci-raw 47.02 (47.03, 47.01, 47.02); gpu-chat-no-think 36.22 (36.22, 36.21, 36.22) | 30 (0) / 4 (0) |
| pr-1 | fibonacci-raw 47.63 (47.62, 47.63, 47.65); gpu-chat-no-think 36.41 (36.41, 36.42, 36.39) | 30 (0) / 4 (0) |
| pr-2 | fibonacci-raw 47.63 (47.63, 47.63, 47.63); gpu-chat-no-think 36.44 (36.45, 36.44, 36.43) | 30 (0) / 4 (0) |
| base-2 | fibonacci-raw 47.00 (47.0, 47.02, 46.97); gpu-chat-no-think 36.18 (36.18, 36.15, 36.18) | 30 (0) / 4 (0) |

| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | bench_cache cold ttft_s (cached tokens, prompt tokens) | every cold ttft_s: measured - logged heat wait = net | qualified |
|---|---|---|---|---|
| base-1 | 47.0, 136.3, 36.2, 108.6 | 4.463 (0, 8186) | 4.463 - 0.0 = 4.463, 6.706 - 4.0 = 2.706, 6.705 - 4.0 = 2.705, 4.452 - 0.0 = 4.452 | True |
| pr-1 | 47.7, 139.0, 36.4, 98.3 | 4.456 (0, 8186) | 4.456 - 0.0 = 4.456, 6.449 - 2.0 = 4.449, 8.439 - 4.0 = 4.439, 8.700 - 6.0 = 2.700 | True |
| pr-2 | 47.6, 138.9, 36.4, 99.2 | 4.463 (0, 8186) | 4.463 - 0.0 = 4.463, 6.443 - 2.0 = 4.443, 8.446 - 4.0 = 4.446, 6.455 - 2.0 = 4.455 | True |
| base-2 | 47.0, 136.3, 36.2, 108.4 | 4.535 (0, 8186) | 4.535 - 0.0 = 4.535, 6.767 - 4.0 = 2.767, 4.513 - 0.0 = 4.513, 8.448 - 4.0 = 4.448 | True |

fibonacci-raw base: median of 6 runs 47.02 tok/s
fibonacci-raw pr: median of 6 runs 47.63 tok/s
gpu-chat-no-think base: median of 6 runs 36.20 tok/s
gpu-chat-no-think pr: median of 6 runs 36.42 tok/s

| run | prompt | drafted tok/s, median of 3 (all) | tokens a round | acceptance | draft:false tok/s | drafted == draft:false, stable |
|---|---|---|---|---|---|---|
| base-1 | repetition | 52.81 (52.80, 52.81, 52.81) | 8.53 | 0.528 | 20.94 | True |
| base-1 | count | 82.46 (82.46, 82.43, 82.46) | 12.90 | 0.953 | 20.91 | True |
| base-1 | tool_json | 62.31 (62.31, 62.35, 62.29) | 5.95 | 0.608 | 20.86 | True |
| pr-1 | repetition | 51.44 (51.44, 51.45, 51.41) | 8.26 | 0.512 | 20.88 | True |
| pr-1 | count | 80.83 (80.87, 80.83, 80.81) | 12.50 | 0.956 | 20.86 | True |
| pr-1 | tool_json | 64.21 (64.22, 64.21, 64.18) | 6.24 | 0.607 | 20.81 | True |
| pr-2 | repetition | 51.40 (51.40, 51.44, 51.39) | 8.26 | 0.512 | 20.90 | True |
| pr-2 | count | 80.88 (80.83, 80.88, 80.88) | 12.50 | 0.956 | 20.88 | True |
| pr-2 | tool_json | 64.23 (64.26, 64.23, 64.22) | 6.24 | 0.607 | 20.81 | True |
| base-2 | repetition | 52.74 (52.75, 52.74, 52.73) | 8.53 | 0.528 | 20.91 | True |
| base-2 | count | 82.38 (82.36, 82.44, 82.38) | 12.90 | 0.953 | 20.88 | True |
| base-2 | tool_json | 62.31 (62.29, 62.32, 62.31) | 5.95 | 0.608 | 20.55 | True |

repetition base: median of 6 runs 52.78 tok/s
repetition pr: median of 6 runs 51.43 tok/s
count base: median of 6 runs 82.43 tok/s
count pr: median of 6 runs 80.85 tok/s
tool_json base: median of 6 runs 62.31 tok/s
tool_json pr: median of 6 runs 64.22 tok/s

| run | ready s | admission |
|---|---|---|
| base-1 | 36.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.97 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 31.4 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-1 | 68.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.2 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-2 | 18.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.1 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-2 | 18.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.97 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 31.5 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |

| rows | base ms | pr ms |
|---|---|---|
| 1 | 46.83 | 46.91 |
| 2 | 54.81 | 54.74 |
| 3 | 60.66 | 62.56 |
| 4 | 66.51 | 68.31 |
| 5 | 71.58 | 74.05 |
| 6 | 76.65 | 79.72 |
| 7 | 81.34 | 85.19 |
| 8 | 86.03 | 90.36 |
| 9 | 90.53 | 95.53 |
| 10 | 94.66 | 100.17 |
| 11 | 98.78 | 104.43 |
| 12 | 102.91 | 108.62 |
| 13 | 107.03 | 112.82 |
| 14 | 111.16 | 116.76 |
| 15 | 115.28 | 120.61 |
| 16 | 117.64 | 124.41 |
| 17 |  | 129.84 |
| 18 |  | 133.71 |
| 19 |  | 136.58 |
| 20 |  | 139.45 |
| 21 |  | 142.32 |
| 22 |  | 144.61 |
| 23 |  | 146.90 |
| 24 |  | 149.18 |
| 25 |  | 151.47 |
| 26 |  | 153.75 |
| 27 |  | 156.04 |
| 28 |  | 158.33 |
| 29 |  | 160.61 |
| 30 |  | 162.90 |
| 31 |  | 165.18 |
| 32 |  | 166.75 |
| level | 3.436 | 3.432 |
