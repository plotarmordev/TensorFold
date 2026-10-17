| run | commit | code token_sha256 (drafted == draft:false) | chat token_sha256 (drafted == draft:false) |
|---|---|---|---|
| base-a | 6ee828c2 | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-a | native-29b20747 | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-b | native-29b20747 | 51e4c630298a14a1 | 161034fd0a090a34 |
| base-b | 6ee828c2 | 51e4c630298a14a1 | 161034fd0a090a34 |

| run | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|
| base-a | fibonacci-raw 42.63 (42.6, 42.63, 42.63); gpu-chat-no-think 32.63 (32.63, 32.63, 32.63) | 30 (0) / 4 (0) |
| pr-a | fibonacci-raw 46.73 (46.72, 46.73, 46.74); gpu-chat-no-think 36.02 (36.02, 36.01, 36.02) | 30 (0) / 4 (0) |
| pr-b | fibonacci-raw 46.84 (46.83, 46.84, 46.87); gpu-chat-no-think 35.75 (35.9, 35.44, 35.75) | 30 (0) / 4 (0) |
| base-b | fibonacci-raw 42.49 (42.49, 42.47, 42.49); gpu-chat-no-think 32.52 (32.52, 32.51, 32.54) | 30 (0) / 4 (0) |

| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | bench_cache cold ttft_s (cached tokens, prompt tokens) | qualified |
|---|---|---|---|
| base-a | 42.7, 125.3, 32.6, 99.2 | 4.463 (0, 8186) | True |
| pr-a | 46.7, 134.5, 36.0, 98.5 | 4.480 (0, 8186) | True |
| pr-b | 46.9, 134.7, 36.1, 98.4 | 4.503 (0, 8186) | True |
| base-b | 42.5, 125.0, 32.5, 99.0 | 4.461 (0, 8186) | True |

fibonacci-raw base: median of 6 runs 42.55 tok/s
fibonacci-raw pr: median of 6 runs 46.78 tok/s
gpu-chat-no-think base: median of 6 runs 32.59 tok/s
gpu-chat-no-think pr: median of 6 runs 35.95 tok/s
