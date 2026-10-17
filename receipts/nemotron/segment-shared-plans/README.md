# Receipts: nemotron/segment-shared-plans

Branch tip 57418b92 against main 6ee828c2. One DGX Spark (GB10) for `segments/` and `plain/`, a second for `speed/`
and `prompt-cuts/`; driver 580.173.02, CUDA 13.0, Ubuntu 24.04.4 LTS; checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`;
`TF_HEAT_HIGH=85 TF_HEAT_LOW=75` on every run. `guard.log` files record GPU temperature and free memory every 5 s.

Binaries (sha256), built with `-Doptimize=fast -Dsm=121` from the same fatbins:

| Build | `tensorfold-native` | `tf-cuda-test` |
|---|---|---|
| main 6ee828c2 | `5af1edaf4157995d8e38df5f831a67b131f416941955420f47a8ee8774d4b09b` | `fef0d4b7d2da4a6e8c45412410eb66384171204c6f732d80ccc9f78d83909997` |
| main + the test alone (57418b92 with `cuda_state.zig` from main) | - | `957b2fc37515230e97ae8db2ae37a6710854fad897b3c4019730895f0560aa4e` |
| this PR 57418b92 | `af7035f2280aab81c45b4829b64b6738924414667b6625d8ace504183fc5a91e` | `5acf20b1576146cfe90fabbf47e8ea89f15bbf6c134308eff30d4d694b266bd8` |

| Path | What |
|---|---|
| `segments/long_document.request.json` | The reproducer: a `/v1/chat/completions` body (8,186 prompt tokens, thinking off, greedy, 256 tokens). |
| `segments/long_document.ids.{txt,json}` | Its rendered ids (`/tokenize` with the same messages), for `tf-cuda-test` and `bench_cache.py --prompt-file`. |
| `segments/` | `serve --context 18432 --parallel 1 --prompt-cache-gib 0 --segments 1\|2`, main then PR: `replies.json`, server logs; `prompt-segments.testonly.log` (FAIL) and `prompt-segments.pr.log` (PASS). |
| `plain/` | main again at `--segments 1` and `2`, drafted and `"draft": false`. |
| `speed/` | default `serve`, boots main, PR, PR, main: `bench_cache.py --require-cold` on the ids, `bench_openai.py --tokens 256 --reps 3 --temperatures 0`, and on the first PR boot `bench_concurrent.py --alone --serial`. |
| `prompt-cuts/` | `tf-cuda-test prompt-cuts` on the ids, PR and main: the same two passes and the same heat-gate failure on both. |
| `host-checks.txt` | `zig build test`, `test-kernel-contract`, `test-golden`, `fuzz-cortheon`, `zig fmt --check`, the lean check. |
