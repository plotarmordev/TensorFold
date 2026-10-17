| run | commit | server sha256 | costs measured fresh | ready s | result |
|---|---|---|---|---|---|
| base-1 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | True | 66.0 | completed |
| pr-1 | f58dc04b4a98ed2e5b89c17bb5966ef58a1df4c7 | 3eefef75610ce5b42262f9c4ad5a347c8d36f04ba62022811edf3afe782a2596 | True | 62.0 | completed |
| pr-2 | f58dc04b4a98ed2e5b89c17bb5966ef58a1df4c7 | 3eefef75610ce5b42262f9c4ad5a347c8d36f04ba62022811edf3afe782a2596 | False | 20.0 | completed |
| base-2 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | False | 22.0 | completed |
| pr-3 | f58dc04b4a98ed2e5b89c17bb5966ef58a1df4c7 | 3eefef75610ce5b42262f9c4ad5a347c8d36f04ba62022811edf3afe782a2596 | False | 24.0 | completed |
| base-3 | fa2af585fc52b888bbba09e379367b8e0792115b | 097904fb69e20ae74986e50ab438db6abec16a68aaa0ce74d51ef4656513179b | False | 22.0 | completed |

| run | admission |
|---|---|
| base-1 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.1 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-1 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.22 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 30.9 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-2 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.22 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.8 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-2 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.0 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| pr-3 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.22 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 31.0 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |
| base-3 | CUDA sm_121 device 0 (NVIDIA GB10, memory shared with the host): model 71.03 GiB; 8 streams at once, 0.40 GiB each at a 16384-token window, of 32.0 GiB left after a 12.2 GiB reserve; prompts in passes of up to 4096 rows, no prompt cache; own glue kernels |

| run | head | level ms | window ms at 1 / 2 / 4 / 8 / 16 / 17 / 24 / 32 rows |
|---|---|---|---|
| base-1 | full | 3.484 | 46.49 / 54.59 / 66.70 / 86.07 / 117.74 / 122.16 / 143.80 / 164.91 |
| pr-1 | full | 3.492 | 46.37 / 54.51 / 66.67 / 86.07 / 118.71 / 122.23 / 143.62 / 164.49 |
| pr-1 | subset | 2.069 | 46.37 / 54.51 / 66.67 / 86.07 / 118.71 / 122.23 / 143.62 / 164.49 |
| pr-2 | full | 3.492 | 46.37 / 54.51 / 66.67 / 86.07 / 118.71 / 122.23 / 143.62 / 164.49 |
| pr-2 | subset | 2.069 | 46.37 / 54.51 / 66.67 / 86.07 / 118.71 / 122.23 / 143.62 / 164.49 |
| base-2 | full | 3.484 | 46.49 / 54.59 / 66.70 / 86.07 / 117.74 / 122.16 / 143.80 / 164.91 |
| pr-3 | full | 3.492 | 46.37 / 54.51 / 66.67 / 86.07 / 118.71 / 122.23 / 143.62 / 164.49 |
| pr-3 | subset | 2.069 | 46.37 / 54.51 / 66.67 / 86.07 / 118.71 / 122.23 / 143.62 / 164.49 |
| base-3 | full | 3.484 | 46.49 / 54.59 / 66.70 / 86.07 / 117.74 / 122.16 / 143.80 / 164.91 |

| run | code / chat token_sha256 (drafted == draft:false) | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |
|---|---|---|---|
| base-1 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 46.80 (46.81, 46.8, 46.79); gpu-chat-no-think 36.04 (36.01, 36.05, 36.04) | 30 (0) / 4 (0) |
| pr-1 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 50.45 (50.47, 50.45, 50.43); gpu-chat-no-think 38.60 (38.61, 38.6, 38.6) | 30 (0) / 4 (0) |
| pr-2 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 50.59 (50.56, 50.6, 50.59); gpu-chat-no-think 38.70 (38.68, 38.72, 38.7) | 30 (0) / 4 (0) |
| base-2 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 46.64 (46.64, 46.43, 46.69); gpu-chat-no-think 35.81 (35.97, 35.81, 35.67) | 30 (0) / 4 (0) |
| pr-3 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 50.47 (50.46, 50.47, 50.49); gpu-chat-no-think 38.58 (38.58, 38.58, 38.63) | 30 (0) / 4 (0) |
| base-3 | 51e4c630298a14a1 / 161034fd0a090a34 | fibonacci-raw 46.61 (46.63, 46.61, 46.59); gpu-chat-no-think 35.89 (35.83, 35.89, 35.89) | 30 (0) / 4 (0) |

| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | every cold ttft_s: measured - logged heat wait = net | qualified |
|---|---|---|---|
| base-1 | 46.8, 138.2, 36.1, 98.2 | 4.471 - 0.0 = 4.471, 6.452 - 2.0 = 4.452, 6.455 - 2.0 = 4.455, 8.455 - 4.0 = 4.455 | True |
| pr-1 | 50.4, 141.2, 38.6, 101.3 | 8.728 - 6.0 = 2.728, 8.466 - 4.0 = 4.466, 8.469 - 4.0 = 4.469, 12.549 - 8.0 = 4.549 | True |
| pr-2 | 50.6, 141.6, 38.7, 100.8 | 4.472 - 0.0 = 4.472, 4.459 - 0.0 = 4.459, 6.461 - 2.0 = 4.461, 6.712 - 4.0 = 2.712 | True |
| base-2 | 46.5, 135.8, 35.9, 96.6 | 4.468 - 0.0 = 4.468, 4.455 - 0.0 = 4.455, 6.459 - 2.0 = 4.459, 6.451 - 2.0 = 4.451 | True |
| pr-3 | 50.5, 141.0, 38.6, 100.3 | 4.461 - 0.0 = 4.461, 4.452 - 0.0 = 4.452, 6.446 - 2.0 = 4.446, 6.450 - 2.0 = 4.450 | True |
| base-3 | 46.6, 134.8, 35.9, 97.7 | 4.477 - 0.0 = 4.477, 4.459 - 0.0 = 4.459, 6.464 - 2.0 = 4.464, 6.464 - 2.0 = 4.464 | True |

fibonacci-raw base: median of 9 runs 46.64 tok/s
fibonacci-raw pr: median of 9 runs 50.47 tok/s
gpu-chat-no-think base: median of 9 runs 35.89 tok/s
gpu-chat-no-think pr: median of 9 runs 38.61 tok/s

| run | prompt | drafted tok/s, median of 3 (all) | tokens a round | acceptance | draft:false tok/s | token_sha256 | drafted == draft:false, stable |
|---|---|---|---|---|---|---|---|
| base-1 | repetition | 52.45 (52.45, 52.45, 52.44) | 8.53 | 0.527 | 20.91 | 3755fae0588986ff | True |
| base-1 | count | 83.16 (83.25, 83.15, 83.16) | 13.33 | 0.963 | 20.89 | 1b3d78d3b1bfd175 | True |
| base-1 | tool_json | 64.94 (64.88, 64.98, 64.94) | 6.40 | 0.608 | 20.83 | 124fa4469513a3d6 | True |
| pr-1 | repetition | 54.54 (54.51, 54.54, 54.54) | 8.26 | 0.502 | 20.93 | 3755fae0588986ff | True |
| pr-1 | count | 96.65 (96.67, 96.56, 96.65) | 13.79 | 0.976 | 20.91 | 1b3d78d3b1bfd175 | True |
| pr-1 | tool_json | 68.76 (68.85, 68.76, 68.69) | 6.40 | 0.617 | 20.86 | 124fa4469513a3d6 | True |
| pr-2 | repetition | 54.61 (54.61, 54.61, 54.66) | 8.26 | 0.502 | 20.99 | 3755fae0588986ff | True |
| pr-2 | count | 96.22 (96.22, 96.26, 96.10) | 13.79 | 0.976 | 20.96 | 1b3d78d3b1bfd175 | True |
| pr-2 | tool_json | 68.86 (68.86, 68.95, 68.86) | 6.40 | 0.617 | 20.91 | 124fa4469513a3d6 | True |
| base-2 | repetition | 52.15 (52.25, 52.15, 52.10) | 8.53 | 0.527 | 20.85 | 3755fae0588986ff | True |
| base-2 | count | 82.83 (82.83, 82.83, 82.51) | 13.33 | 0.963 | 20.78 | 1b3d78d3b1bfd175 | True |
| base-2 | tool_json | 64.54 (64.54, 64.50, 64.74) | 6.40 | 0.608 | 20.81 | 124fa4469513a3d6 | True |
| pr-3 | repetition | 54.59 (54.65, 54.52, 54.59) | 8.26 | 0.502 | 20.90 | 3755fae0588986ff | True |
| pr-3 | count | 96.83 (96.72, 96.94, 96.83) | 13.79 | 0.976 | 20.86 | 1b3d78d3b1bfd175 | True |
| pr-3 | tool_json | 68.74 (68.49, 68.77, 68.74) | 6.40 | 0.617 | 20.81 | 124fa4469513a3d6 | True |
| base-3 | repetition | 52.05 (52.05, 52.04, 52.07) | 8.53 | 0.527 | 20.82 | 3755fae0588986ff | True |
| base-3 | count | 82.47 (82.54, 82.42, 82.47) | 13.33 | 0.963 | 20.80 | 1b3d78d3b1bfd175 | True |
| base-3 | tool_json | 64.54 (64.54, 64.65, 64.46) | 6.40 | 0.608 | 20.72 | 124fa4469513a3d6 | True |


| prompt | base median tok/s (n) | PR median tok/s (n) | change | base tokens a round | PR tokens a round | hashes equal across builds |
|---|---|---|---|---|---|---|
| repetition | 52.15 (9) | 54.59 (9) | +4.7% | 8.53 | 8.26 | True |
| count | 82.83 (9) | 96.65 (9) | +16.7% | 13.33 | 13.79 | True |
| tool_json | 64.65 (9) | 68.77 (9) | +6.4% | 6.40 | 6.40 | True |
