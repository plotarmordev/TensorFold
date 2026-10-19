# Receipts: shared windows run the latent shared expert on the side stream again

This pull request is one commit on `main`. Its `zig/` tree equals v1.1.0 (`76aafcf274e60a5e31a59ce145b2f71f7b366e9d`)
plus that commit: `main`'s three commits after the tag touch only `app/`, `docs/` and `README.md`.

The GPU runs below were made before `main` moved past v1.1.0, with builds from clean detached checkouts:

| Build | Source | `tensorfold-native` sha256 | `tf-cuda-test` sha256 | `tensorfold` sha256 |
|---|---|---|---|---|
| base | v1.1.0 `76aafcf2` | `04167a172c847952737a4b401fb5945b74d598a51312ed19f4e6ddd974015088` | — | — |
| branch | v1.1.0 + this commit + the confidence-copy commit (`gpu/measured-branch.diff`) | `ec0ff22f7bdb5354c3faa8d5e1db125fcfbbcbf329df70042f3ca25b02789c69` | `d3e12ff9784c127181a35bb5aba18b7161a2cb19ec2d8d5f2d5b72bb6b0c0e13` | `993cfbbb96033f5af6c24c41404693d549364d522c9bbdd47ce63715f68630bc` |
| testonly | v1.1.0 + `gpu/test-only.diff` (this commit's tests, not its source line) | — | `e44e328b24070bde182da92945e5eb79533ddf5e5a0c905efba2b3ba6cffbd0d` | — |
| mutant | branch + `gpu/mutant-no-join.diff` (the main stream no longer waits for the side stream) | — | `d257ba2bf4adee10a095d11cb8a50b77d71afded1941b59a6c7190a3d1ae76ed` | — |

The branch build also carries the separate confidence-copy commit (its own pull request). It changes only
`cuda_mtp.zig`, `cuda_mtp_policy.zig` and a host test. Without `--mtp-confidence`, which no run here sets, it adds up to
8 small device-to-host copies and one event record after each batched MTP draft. Builds: zig 0.17.0, nvcc 13.0
(r13.0) fatbins for sm_121, `zig build native -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline -Doptimize=fast -Dstrip=true
-Dversion=1.1.0`, `zig build install` with the same flags and that build's fatbins. Hardware: two DGX Spark (GB10)
nodes, `<gb10-a>` and `<gb10-b>`, driver 580.173.02 (CUDA 13.0). Checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`. Every run sets
`TF_HEAT_HIGH=85 TF_HEAT_LOW=75`; `*.guard.log` records GPU and hottest thermal-zone temperature, load, free memory and swap every 5 s.
`*.hold.log` starts with the node, the binary's sha256 and the command line. Private paths, host names and addresses are
placeholders (`<home>`, `<checkpoint>`, `<gb10-a>`, `<loopback>`, …), and one internal benchmark label is renamed
`wave` in the client and its output; nothing else differs from the files the runs wrote.

| Path | What |
|---|---|
| `host/` | This commit's tip and its base (`main`), each step on a fresh cache: `scripts/host-tests.sh` (`zig build test-kernel-contract`, `test`, `test-golden`, `fuzz-cortheon`, extra steps, `zig fmt --check zig build.zig`, `tools/zig/lean_check.py`). `before.*`: the tip with only the source line reverted (`host/before.diff`), `zig build test-nemotron-mtp-policy-host`. `native.txt`: the tip's release-flag `tensorfold-native` (`scripts/native.sh`, fatbins from `zig build fatbins -Dsm=121 -Dnvcc=…`). `fatbins.txt`: those fatbins against the measured branch build's, 83 of 84 byte-equal; `glm_quant.fatbin` differs among all three, from the same kernel source. `lean-new.txt`: the tip's lean problems not in the base's (none). |
| `gpu/shared-fork.{branch,testonly,mutant}.*` | `tf-cuda-test shared-fork <checkpoint>` on `<gb10-b>`, each build. |
| `gpu/shared-widths-{2,4,8}.branch.*` | `tensorfold shared-widths <checkpoint> <ids>/prompts-ids.json --streams N --max-tokens 400`: every kept row of N concurrent streams against the stream alone. |
| `tp2/cached-plans-tp2.branch.rank{0,1}.*` | `tf-cuda-test cached-plans-tp2` across the two nodes (rank 0 on `<gb10-a>`), NCCL 2.28.9. |
| `tp2/serve.branch.txt`, `tp2/serve.stock-1.1.0.txt` | Two-rank serving (`serve --parallel 1 --prompt-cache-gib 0`, NCCL 2.32.3, rank 0 in a fresh `HOME`) and `scripts/bench.py decode`: drafted against `"draft": false` per prompt. The harness's own 8-minute limit stopped the branch's server during the code requests, after three client pauses for the node to cool (`cooling:` lines); the stock release's bench finished. |
| `speed/a/{1-base,2-branch,3-base}.*` | `<gb10-a>`, in this order, each a fresh server and a fresh `HOME` (so each times its own window costs): `serve <checkpoint> --context 131072 --parallel auto --vision --thinking`; `scripts/waves.py` count and prose at 2, 4 and 8 streams; then `scripts/bench.py streams`. |
| `speed/b/{1-base,2-branch,3-base}.*` | `<gb10-b>`, the same serve flags and order: `waves.py` count, prose and code at 4 and 8 streams; then `bench.py decode count prose code` (one stream, drafted against `"draft": false`). |
| `scripts/modes.py` | The like-mode table in the pull request: `python3 scripts/modes.py speed/a speed/b`. |
| `scripts/` | Also `waves.py`, `bench.py`, `prompts.py` (the client), `host-tests.sh`, `step.sh`. |

`waves.py`: per case, one reply alone (the reference), then a warmup wave and 3 measured waves of N identical
requests released together; 400 output tokens each (`ignore_eos`), greedy, thinking off, streamed. `wave_tps` is all
replies' output tokens over dispatch to the last token; each wave records every reply's sha256 against the reply alone.
N identical requests run in one of two modes: request 2 is admitted one prompt pass after request 1 (on time), or one
lone round later (late, about 240 ms of lag instead of about 90). On count, late waves are about 20 % slower on every
build, the base included (the profiled difference is in the routed-expert kernels). `modes.py` splits waves by that lag and compares the base and the branch in the same mode.
