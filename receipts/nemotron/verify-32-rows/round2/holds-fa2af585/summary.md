| run | commit | server sha256 | code token_sha256 (drafted == draft:false) | chat token_sha256 (drafted == draft:false) |
|---|---|---|---|---|
| base-1 | 39993136981823d314a1a6204ecc6f95ae25bb45 | ef46e9ba4e9b05a763306f3c37ef99ec7bf77892ec657bb078ac37e3b49c6e7c | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-1 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-2 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | 51e4c630298a14a1 | 161034fd0a090a34 |
| base-2 | 39993136981823d314a1a6204ecc6f95ae25bb45 | ef46e9ba4e9b05a763306f3c37ef99ec7bf77892ec657bb078ac37e3b49c6e7c | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-3 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | 51e4c630298a14a1 | 161034fd0a090a34 |
| base-3 | 39993136981823d314a1a6204ecc6f95ae25bb45 | ef46e9ba4e9b05a763306f3c37ef99ec7bf77892ec657bb078ac37e3b49c6e7c | 51e4c630298a14a1 | 161034fd0a090a34 |

| run | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|
| base-1 | fibonacci-raw 46.80 (46.82, 46.79, 46.8); gpu-chat-no-think 36.13 (36.13, 36.13, 36.14) | 30 (0) / 4 (0) |
| pr-1 | fibonacci-raw 46.84 (46.84, 46.84, 46.87); gpu-chat-no-think 36.21 (36.21, 36.21, 36.19) | 30 (0) / 4 (0) |
| pr-2 | fibonacci-raw 46.81 (46.8, 46.81, 46.84); gpu-chat-no-think 36.02 (36.15, 36.02, 35.61) | 30 (0) / 4 (0) |
| base-2 | fibonacci-raw 46.87 (46.86, 46.88, 46.87); gpu-chat-no-think 36.16 (36.18, 36.16, 36.16) | 30 (0) / 4 (0) |
| pr-3 | fibonacci-raw 46.65 (46.65, 46.66, 46.64); gpu-chat-no-think 36.05 (36.05, 36.06, 36.05) | 30 (0) / 4 (0) |
| base-3 | fibonacci-raw 46.65 (46.63, 46.65, 46.66); gpu-chat-no-think 36.03 (36.04, 36.02, 36.03) | 30 (0) / 4 (0) |

| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | bench_cache cold ttft_s (cached tokens, prompt tokens) | every cold ttft_s: measured - logged heat wait = net | qualified |
|---|---|---|---|---|
| base-1 | 46.8, 135.5, 36.1, 98.5 | 4.465 (0, 8186) | 4.465 - 0.0 = 4.465, 8.450 - 4.0 = 4.450, 8.453 - 4.0 = 4.453, 8.451 - 4.0 = 4.451 | True |
| pr-1 | 46.8, 138.8, 36.2, 98.3 | 4.461 (0, 8186) | 4.461 - 0.0 = 4.461, 4.450 - 0.0 = 4.450, 6.705 - 4.0 = 2.705, 4.452 - 0.0 = 4.452 | True |
| pr-2 | 46.8, 138.6, 36.2, 98.2 | 6.465 (0, 8186) | 6.465 - 2.0 = 4.465, 8.459 - 4.0 = 4.459, 8.446 - 4.0 = 4.446, 8.446 - 4.0 = 4.446 | True |
| base-2 | 46.9, 136.2, 36.2, 98.5 | 4.465 (0, 8186) | 4.465 - 0.0 = 4.465, 8.449 - 4.0 = 4.449, 8.451 - 4.0 = 4.451, 8.453 - 4.0 = 4.453 | True |
| pr-3 | 46.7, 138.1, 36.1, 98.2 | 4.470 (0, 8186) | 4.470 - 0.0 = 4.470, 4.461 - 0.0 = 4.461, 6.457 - 2.0 = 4.457, 6.462 - 2.0 = 4.462 | True |
| base-3 | 46.7, 135.8, 36.0, 98.4 | 6.469 (0, 8186) | 6.469 - 2.0 = 4.469, 8.454 - 4.0 = 4.454, 8.456 - 4.0 = 4.456, 8.453 - 4.0 = 4.453 | True |

fibonacci-raw base: median of 9 runs 46.80 tok/s
fibonacci-raw pr: median of 9 runs 46.81 tok/s
gpu-chat-no-think base: median of 9 runs 36.13 tok/s
gpu-chat-no-think pr: median of 9 runs 36.06 tok/s

| run | prompt | drafted tok/s, median of 3 (all) | tokens a round | acceptance | draft:false tok/s | drafted == draft:false, stable |
|---|---|---|---|---|---|---|
| base-1 | repetition | 52.75 (52.74, 52.80, 52.75) | 8.53 | 0.528 | 20.94 | True |
| base-1 | count | 80.01 (80.01, 80.01, 79.99) | 12.50 | 0.929 | 20.90 | True |
| base-1 | tool_json | 62.29 (62.29, 62.21, 62.32) | 5.95 | 0.608 | 20.84 | True |
| pr-1 | repetition | 52.60 (52.61, 52.58, 52.60) | 8.53 | 0.527 | 20.91 | True |
| pr-1 | count | 83.59 (83.59, 83.66, 83.57) | 13.33 | 0.963 | 20.89 | True |
| pr-1 | tool_json | 65.07 (65.05, 65.07, 65.13) | 6.40 | 0.608 | 20.84 | True |
| pr-2 | repetition | 52.00 (52.06, 51.72, 52.00) | 8.53 | 0.527 | 20.87 | True |
| pr-2 | count | 83.59 (83.59, 83.62, 83.59) | 13.33 | 0.963 | 20.84 | True |
| pr-2 | tool_json | 65.10 (65.13, 65.10, 65.07) | 6.40 | 0.608 | 20.79 | True |
| base-2 | repetition | 52.78 (52.78, 52.78, 52.80) | 8.53 | 0.528 | 20.96 | True |
| base-2 | count | 80.25 (80.25, 80.25, 80.27) | 12.50 | 0.929 | 20.94 | True |
| base-2 | tool_json | 62.33 (62.33, 62.36, 62.33) | 5.95 | 0.608 | 20.89 | True |
| pr-3 | repetition | 52.36 (52.36, 52.35, 52.36) | 8.53 | 0.527 | 20.91 | True |
| pr-3 | count | 83.01 (82.99, 83.01, 83.10) | 13.33 | 0.963 | 20.89 | True |
| pr-3 | tool_json | 64.91 (64.91, 64.87, 64.91) | 6.40 | 0.608 | 20.85 | True |
| base-3 | repetition | 52.55 (52.55, 52.56, 52.52) | 8.53 | 0.528 | 20.94 | True |
| base-3 | count | 79.71 (79.68, 79.72, 79.71) | 12.50 | 0.929 | 20.92 | True |
| base-3 | tool_json | 62.16 (62.13, 62.16, 62.20) | 5.95 | 0.608 | 20.87 | True |

repetition base: median of 9 runs 52.75 tok/s
repetition pr: median of 9 runs 52.36 tok/s
count base: median of 9 runs 80.01 tok/s
count pr: median of 9 runs 83.59 tok/s
tool_json base: median of 9 runs 62.29 tok/s
tool_json pr: median of 9 runs 65.07 tok/s

| run | ready s | admission |
|---|---|---|
| base-1 | 38.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.97 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 31.1 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-1 | 66.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.1 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-2 | 20.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.0 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-2 | 20.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.97 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 32.4 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-3 | 20.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.0 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-3 | 20.0 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 70.97 GiB; 8 streams at once, 0.35 GiB each at a 16384-token window, of 31.5 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |

| rows | base ms | pr ms |
|---|---|---|
| 1 | 46.86 | 46.86 |
| 2 | 54.88 | 54.55 |
| 3 | 60.90 | 60.63 |
| 4 | 66.91 | 66.71 |
| 5 | 71.83 | 71.71 |
| 6 | 76.74 | 76.71 |
| 7 | 81.37 | 81.37 |
| 8 | 85.99 | 86.03 |
| 9 | 90.54 | 90.56 |
| 10 | 94.70 | 94.70 |
| 11 | 98.86 | 98.83 |
| 12 | 103.02 | 102.97 |
| 13 | 107.18 | 107.11 |
| 14 | 111.34 | 111.25 |
| 15 | 115.49 | 115.38 |
| 16 | 117.69 | 117.69 |
| 17 |  | 122.12 |
| 18 |  | 125.41 |
| 19 |  | 128.47 |
| 20 |  | 131.53 |
| 21 |  | 134.59 |
| 22 |  | 137.66 |
| 23 |  | 140.72 |
| 24 |  | 143.78 |
| 25 |  | 146.84 |
| 26 |  | 149.90 |
| 27 |  | 152.96 |
| 28 |  | 155.36 |
| 29 |  | 157.76 |
| 30 |  | 160.16 |
| 31 |  | 162.54 |
| 32 |  | 164.80 |
| level | 3.463 | 3.433 |
