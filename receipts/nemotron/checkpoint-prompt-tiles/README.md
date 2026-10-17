# Receipts: nemotron/checkpoint-prompt-tiles

Branch tip `a79678c5624e55f5b009700e4d953e8034de9336` against its base `50a34190b04055ff9268cc12f9d8284323830a25`
(the merge of #999 and #1002 it sits on). Every binary below was built from a clean checkout of that commit
(`git worktree add --detach`), the mutant from a clean checkout of the tip plus `gpu/mutant.diff`. Cold prefill holds on
one DGX Spark (GB10); decode holds and `tf-cuda-test` on a second; driver 580.173.02, CUDA 13.0 (nvcc 13.0.r13.0);
checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`;
`TF_HEAT_HIGH=85 TF_HEAT_LOW=75` on every run. `*.guard.jsonl` record GPU temperature and free memory every 5 s.

`builds.txt`: each build's source commit and `git status`, the build commands (zig 0.17.0) and the full sha256 of every
binary used:

| Build | Source | `tensorfold-native` sha256 | `tf-cuda-test` sha256 |
|---|---|---|---|
| base | `50a34190b04055ff9268cc12f9d8284323830a25` | `38cb28f3e07de241d172d7579c9b1b55a291bdfc45475506c2e5a7c443236dc4` | `08662952ea63e65d7733aea06130985467b17deee4fbcfe51d23823a28935e62` |
| tip | `a79678c5624e55f5b009700e4d953e8034de9336` | `692e2fd510491ae264fac0deaa66b6453eacc367adb50badeda332a3d1279d17` | `68646553a7b3e3cec21efd411bd3940102ea68966f7b33d460e8ec1158a51dc2` |
| mutant | tip + `gpu/mutant.diff` | - | `07ffd8260988b09c1857aa10318f93a485701023301caa8b27636bb401bc22d7` |

Every hold is a fresh server. Each binary's window-cost cache entries were deleted before its first boot on each node
(`costs_cache_deleted_before_boot` in the hold's `.json`), so it timed its own table. Every JSON a hold writes carries
`source_commit` and `binary_sha256`, and the hold's `.json` also the build commands. Every `gpu/*.log` starts with its
source commit and `tf-cuda-test` sha256. Paths private to the machines are placeholders (`<home>`, `<loopback>`,
`<cache>`).

| Path | What |
|---|---|
| `host/{tip,base}.txt` (+ `.*.log`, `.lean.txt`), `host/lean-new.txt` | On the clean checkouts, each step with a fresh cache: `zig build test-kernel-contract`, `test`, `test-golden`, `fuzz-cortheon`, `zig fmt --check zig build.zig`, `tools/zig/lean_check.py`; `lean-new.txt` lists the tip's problems not in the base's set (none). |
| `gpu/{checkpoint,glue,modelopt,kernel-contract,kernel-contract-limits}.tip.log` | `tf-cuda-test` on the tip; `kernel-contract-limits` reads the sm_80/86/89/120/121 fatbins of the same checkout (`zig build fatbins kernel-contract-fatbins -Dsm=80,86,89,120,121`). |
| `gpu/{checkpoint,glue,kernel-contract}.mut.log`, `gpu/mutant.diff` | The same tests on the tip with two planted changes: the prompt down rounds share 0's sum to bf16 before adding share 1, and the router tiles take each word's high input before its low one. |
| `decode/{base-d,pr-d,pr-d2,base-d2}.*` | `serve MODEL --name super --no-update-check --prompt-cache-gib 0`, in this order: drafted against `"draft": false` per prompt (`.json` `checks`), `bench_concurrent.py --levels 1,4 --tokens 256 --reps 3 --temperatures 0 --alone --serial`, `bench_openai.py --tokens 256 --reps 3 --temperatures 0`. |
| `prefill/{base-1,pr-1,pr-2,base-2}.*` | The same serve flags plus `--context 131584`, in this order: for each token array of 2,048, 8,192, 32,768 and 65,536 ids (`ids_sha256` in the `.json`), `bench_cache.py --require-cold --tokens 16 --repeats 1 --streams 1 --expected-prompt-tokens N` (three cold requests), then the array with `"draft": false` against the drafted reply. |
| `prefill/{base-L1,pr-L1,pr-L2,base-L2}.*` | The same at 131,072 ids. |
| `summary.md` | `summarize.py` over the holds: build and admission per hold, decode tables, every cold `ttft_s` minus the heat waits the server logged inside that request, and medians base against tip. |
| `scripts/` | `build.sh`, `host-tests.sh`, `gpu-tests.sh`, `holds.sh`, `hold.py`, `summarize.py`. |
