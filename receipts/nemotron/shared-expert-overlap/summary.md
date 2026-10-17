| run | commit | code token_sha256 (drafted == draft:false) | chat token_sha256 (drafted == draft:false) |
|---|---|---|---|
| base-1 | ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-1 | 14c78e735cc9effb2dfb2c2dc53c54b93d06f481 | 51e4c630298a14a1 | 161034fd0a090a34 |
| pr-2 | 14c78e735cc9effb2dfb2c2dc53c54b93d06f481 | 51e4c630298a14a1 | 161034fd0a090a34 |
| base-2 | ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba | 51e4c630298a14a1 | 161034fd0a090a34 |

| run | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|
| base-1 | fibonacci-raw 46.72 (46.72, 46.72, 46.72); gpu-chat-no-think 36.06 (36.06, 36.05, 36.06) | 30 (0) / 4 (0) |
| pr-1 | fibonacci-raw 47.85 (47.75, 47.89, 47.85); gpu-chat-no-think 36.78 (36.79, 36.78, 36.75) | 30 (0) / 4 (0) |
| pr-2 | fibonacci-raw 47.93 (47.93, 47.93, 47.91); gpu-chat-no-think 36.82 (36.82, 36.82, 36.81) | 30 (0) / 4 (0) |
| base-2 | fibonacci-raw 46.69 (46.68, 46.7, 46.69); gpu-chat-no-think 36.06 (36.06, 36.06, 36.06) | 30 (0) / 4 (0) |

| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | bench_cache cold ttft_s (cached tokens, prompt tokens) | qualified |
|---|---|---|---|
| base-1 | 46.7, 135.8, 36.1, 108.3 | 4.454 (0, 8186) | True |
| pr-1 | 47.8, 138.2, 36.8, 100.4 | 4.452 (0, 8186) | True |
| pr-2 | 47.9, 138.3, 36.8, 100.4 | 4.466 (0, 8186) | True |
| base-2 | 46.7, 135.8, 36.1, 108.2 | 4.461 (0, 8186) | True |

fibonacci-raw base: median of 6 runs 46.71 tok/s
fibonacci-raw pr: median of 6 runs 47.90 tok/s
gpu-chat-no-think base: median of 6 runs 36.06 tok/s
gpu-chat-no-think pr: median of 6 runs 36.80 tok/s
