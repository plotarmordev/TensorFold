| run | commit | code token_sha256 (drafted == draft:false) | chat token_sha256 (drafted == draft:false) |
|---|---|---|---|
| base-1 | 6ee828c2 | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-1 | native-b8613a54 | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-2 | native-b8613a54 | 51e4c630298a14a1 | 161034fd0a090a34 |
| base-2 | 6ee828c2 | 51e4c630298a14a1 | 161034fd0a090a34 |

| run | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|
| base-1 | fibonacci-raw 42.59 (42.59, 42.59, 42.6); gpu-chat-no-think 32.59 (32.6, 32.59, 32.58) | 30 (0) / 4 (0) |
| pr-1 | fibonacci-raw 46.95 (46.93, 46.95, 46.97); gpu-chat-no-think 36.11 (36.11, 36.12, 36.08) | 30 (0) / 4 (0) |
| pr-2 | fibonacci-raw 47.06 (47.06, 47.06, 47.06); gpu-chat-no-think 36.20 (36.2, 36.19, 36.21) | 30 (0) / 4 (0) |
| base-2 | fibonacci-raw 42.46 (42.46, 42.47, 42.46); gpu-chat-no-think 32.50 (32.51, 32.5, 32.5) | 30 (0) / 4 (0) |

| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | bench_cache cold ttft_s (cached tokens, prompt tokens) | qualified |
|---|---|---|---|
| base-1 | 42.6, 125.1, 32.6, 99.0 | 4.460 (0, 8186) | True |
| pr-1 | 46.9, 134.5, 36.1, 98.5 | 6.464 (0, 8186) | True |
| pr-2 | 47.1, 134.9, 36.2, 98.5 | 4.460 (0, 8186) | True |
| base-2 | 42.5, 125.0, 32.5, 99.0 | 4.465 (0, 8186) | True |

fibonacci-raw base: median of 6 runs 42.53 tok/s
fibonacci-raw pr: median of 6 runs 47.02 tok/s
gpu-chat-no-think base: median of 6 runs 32.55 tok/s
gpu-chat-no-think pr: median of 6 runs 36.16 tok/s
