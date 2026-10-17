# Receipts: nemotron/shared-expert-overlap

Branch tip `14c78e735cc9effb2dfb2c2dc53c54b93d06f481` against its base `ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba`
(the merge of #996 and #999 it sits on). Every binary below was built from a clean checkout of that commit
(`git worktree add --detach`), the two diagnostic builds from a clean checkout plus the one diff named. Serve holds on
one DGX Spark (GB10), `tf-cuda-test` runs on a second; driver 580.173.02, CUDA 13.0 (nvcc 13.0.r13.0); checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`;
`TF_HEAT_HIGH=85 TF_HEAT_LOW=75` on every run. `*.guard.jsonl` record GPU temperature and free memory every 5 s.

`builds.txt`: each build's source commit and `git status`, the three build commands (`build-clean.sh`, zig 0.17.0),
and the full sha256 of every binary used:

| Build | Source | `tensorfold-native` sha256 | `tf-cuda-test` sha256 |
|---|---|---|---|
| base | `ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba` | `f24658bd2f308a05495928e18f9d9e8240323bc11f4573feaef35deaefbdb93f` | `cc76592e24263fb8e3e5ca531a6339011aa2245f0f942fabc9f50b1900e7d4ef` |
| tip | `14c78e735cc9effb2dfb2c2dc53c54b93d06f481` | `33d9dcf6e0064903ce08f723fd6b3559f8b3f03828e71200c9c02e5a314e8d6f` | `07ad762d349ff5028dea72e24327fff73d5c7475a6b0b11919f3ee7431e6710e` |
| base + the test alone | base + `gpu/testonly.diff` | - | `4e044f6e5798735f113305eb1d13880b15bca026b3294832564fe1345b9c6c90` |
| tip without the join | tip + `gpu/mutant-no-join.diff` | - | `298197a24c81775c789fe52f26829590cef6f43cb7fa0f4a49256e41b7818d32` |

Every JSON a hold writes carries `source_commit` and `binary_sha256` (`qualify.py` and `cold.py` add them to each
client's output file right after the client exits; the `--label` the clients record is the hold's label). Every
`gpu/*.log` starts with its source commit and `tf-cuda-test` sha256. Paths private to the machines are placeholders
(`<home>`, `<loopback>`, `<prompts.json>`, `<gpu.lock>`).

| Path | What |
|---|---|
| `{base-1,pr-1,pr-2,base-2}.*` | `serve MODEL --name super --no-update-check --prompt-cache-gib 0`, a fresh server per hold in this order: drafted against `"draft": false` (`.json` `checks`), `bench_concurrent.py --levels 1,4 --tokens 256 --reps 3 --temperatures 0 --alone --serial`, `bench_openai.py --tokens 256 --reps 3 --temperatures 0`, `bench_cache.py --require-cold --tokens 16 --repeats 2 --streams 1` on `*.long_document.ids.json` (8,186 tokens); server logs (`qualify.py`, `holds.sh`). |
| `summary.md` | `summarize.py . base-1 pr-1 pr-2 base-2`. |
| `cold/{pr-3,base-3,base-4,pr-4}.*` | Four more holds in this order, each only the cold `bench_cache.py` request on the same ids (`cold.py`). |
| `gpu/shared-fork.{tip,testonly,mut}.log` | `tf-cuda-test shared-fork MODEL`: PASS on the tip, FAIL with the test alone on the base, FAIL with the join removed (`gpu-tests.sh`). |
| `gpu/window-profile.{base,tip}.log` | `tf-cuda-test window-profile MODEL 1,4,16 gpu/long_document.ids.txt`, base then tip: graph ms and eager ms by kernel class, medians of 5. |
| `host-{base,tip}.txt`, `lean-{base,tip}.txt` | On the clean checkouts, each with a fresh cache: `zig build test-kernel-contract`, `test`, `test-golden`, `fuzz-cortheon`, `zig fmt --check zig build.zig`, `tools/zig/lean_check.py`. |
| `earlier-builds/` | The same four holds run first with earlier builds of the same two commits (worktrees at those commits, not clean checkouts; `tensorfold-native` sha256 base `9336959220ae5ce92b0e0d5f4c2f4ce6c14b8e20161082edfac66de15004404e`, tip `077831bc942fdf1e532e24a759b4c0795b0bc3bc05afe2b6ae954e7e474b3d69`). Kept only for the chat x4 cell's spread; nothing in the description rests on them otherwise. |
