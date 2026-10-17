| run | commit | code token_sha256 (drafted == draft:false) | chat token_sha256 (drafted == draft:false) |
|---|---|---|---|
| base-1 | ed44eefb | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-1 | 14c78e73 | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-2 | 14c78e73 | 51e4c630298a14a1 | 161034fd0a090a34 |
| base-2 | ed44eefb | 51e4c630298a14a1 | 161034fd0a090a34 |

| run | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|
| base-1 | fibonacci-raw 46.89 (46.92, 46.89, 46.89); gpu-chat-no-think 36.25 (36.25, 36.26, 36.23) | 30 (0) / 4 (0) |
| pr-1 | fibonacci-raw 47.58 (47.57, 47.61, 47.58); gpu-chat-no-think 36.76 (36.79, 36.72, 36.76) | 30 (0) / 4 (0) |
| pr-2 | fibonacci-raw 47.76 (47.76, 47.73, 47.87); gpu-chat-no-think 36.85 (36.84, 36.86, 36.85) | 30 (0) / 4 (0) |
| base-2 | fibonacci-raw 46.87 (46.87, 46.87, 46.86); gpu-chat-no-think 36.15 (36.15, 36.13, 36.19) | 30 (0) / 4 (0) |

| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | bench_cache cold ttft_s (cached tokens, prompt tokens) | qualified |
|---|---|---|---|
| base-1 | 46.9, 136.3, 36.2, 98.6 | 4.457 (0, 8186) | True |
| pr-1 | 47.7, 138.3, 36.8, 108.5 | 4.469 (0, 8186) | True |
| pr-2 | 47.7, 138.3, 36.9, 101.1 | 4.482 (0, 8186) | True |
| base-2 | 46.8, 138.2, 36.2, 98.5 | 4.459 (0, 8186) | True |

fibonacci-raw base: median of 6 runs 46.88 tok/s
fibonacci-raw pr: median of 6 runs 47.67 tok/s
gpu-chat-no-think base: median of 6 runs 36.21 tok/s
gpu-chat-no-think pr: median of 6 runs 36.81 tok/s
