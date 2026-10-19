# TP=2 Nemotron 3.5 Super VL with NCCL 2.28.9: logs

Two DGX Spark (GB10) nodes over RoCE, driver 580.173.02 (CUDA 13.0),
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`.
Rank 1 started first, rank 0 about 10 s later with
`serve … --parallel 1 --prompt-cache-gib 0`. Runs 01-17 ran under `gdb -batch -x stall.gdb` on rank 0 (and on rank 1
where a rank-1 stack is listed); a rank 0 not serving after 6 minutes (11 in run 14) got SIGINT on both ranks.

Per run: `NN-name.rank0.log` / `.rank1.log` (server output, NCCL INFO, stacks), `.rank0.launch.log` /
`.rank1.launch.log` (binary sha256, environment and command line), `.client.log` where a client ran. Host names,
addresses and paths are replaced: `rank0-host`, `rank1-host`, 192.0.2.1 / 192.0.2.2, `/opt/nccl-VERSION` (the
directory holding that `libnccl.so.2`). Where a launch line sets `LD_LIBRARY_PATH` twice, the second one is in effect
(each log's `NCCL version` line shows which).

`libnccl.so.2` sha256: 2.28.9 (`nvidia-nccl-cu13` 2.28.9 wheel, `2.28.9+cuda13.0`)
`ad593e48bb6240e88426297db30a2b04a43712ed479064ac0691e27f5d7f4a25`; 2.32.3 (`nvidia-nccl-cu13` 2.32.3 wheel,
`2.32.3+cuda13.4`) `c77f5ced2c4b538aad2b0ab88a9838e9eac9a68e22017863ffb0cff9c4f58745`.

Binaries (`tensorfold-native` sha256):

| name | build | sha256 |
|---|---|---|
| 1.0.12-release | 1.0.12 release archive | `0818b2ad84f517a3aed17cd3c5ea330e59ddb5b2d02484efca3f7deb8dc2db2c` |
| 1.1.0-release | 1.1.0 release archive | `4a6ffec3ca4bfd50ecd3f2948c1ca7d11b62b40269bb9decddd0768558965b3a` |
| 1.1.0-source | v1.1.0 (76aafcf2) from source, `-Doptimize=fast -Dstrip=true` | `04167a172c847952737a4b401fb5945b74d598a51312ed19f4e6ddd974015088` |
| 1.1.0-local-a/b/c | three local branches on v1.1.0, stripped (not upstream code) | `248f34e8…`, `e3c965bb…`, `b6053b8c…` |
| 1.1.0-debug | v1.1.0 from source, `-Doptimize=fast -Dstrip=false` | `3c233f908b20a5f42479aa0a1b898c8fabd5cb07c3c5d6e909ce1730ffc11a61` |
| 1.1.0-no-tp-graphs | v1.1.0 plus `no-tp-graphs.diff` (diagnostic: no captured graphs at TP), unstripped | `af6ba0621211e4ee56e4e2d11540aaf671e2f83b68f09e30034fae160ada0568` |

Runs. "Pair" means a thread spinning in `ncclLocalOpAppend` (`proxy.cc:499`) while the proxy progress thread sleeps in
`ncclProxyGetPostedOps` (`proxy.cc:797`). Width is the window the startup cost timing was on (unstripped builds only).

| run | NCCL | env beyond the launch | rank 0 HOME | result | rank 0 stack | rank 1 stack |
|---|---|---|---|---|---|---|
| 01-1.0.12-release | 2.28.9 | — | new, empty | not serving | pair | — |
| 02-1.1.0-release | 2.28.9 | — | new, empty | not serving | progress thread active (`proxy.cc:986`), main in `cuStreamSynchronize` | — |
| 03-1.1.0-source | 2.28.9 | — | new, empty | not serving | pair | — |
| 04-, 05-, 06-1.1.0-local-a/b/c | 2.28.9 | — | new, empty | not serving | pair | — |
| 07-1.1.0-debug | 2.28.9 | — | new, empty | not serving | pair; `cuda_costs.zig:30`, width 11 | — |
| 08-…graph-register-0 | 2.28.9 | `NCCL_GRAPH_REGISTER=0` | new, empty | not serving | pair; width 8 | — |
| 09-…p2p-ll-threshold-0 | 2.28.9 | `NCCL_P2P_LL_THRESHOLD=0` | new, empty | not serving | pair; width 11 | `cuda_worker.zig:172` |
| 10-…ib-relaxed-ordering-0 | 2.28.9 | `NCCL_IB_PCI_RELAXED_ORDERING=0` | new, empty | not serving | pair; width 12 | `cuda_worker.zig:172` |
| 11-…net-socket | 2.28.9 | `NCCL_NET=Socket` | new, empty | not serving | pair; width 9 | `cuda_worker.zig:172` |
| 12-…proxy-dump-signal | 2.28.9 | `NCCL_PROXY_DUMP_SIGNAL=10` | new, empty | not serving | pair; width 9 | `cuda_worker.zig:172` |
| 13-1.1.0-no-tp-graphs | 2.28.9 | — | new, empty | not serving | cost timing, width 26 (main in `cuLaunchKernel`) | pair, from `ncclSend` (`cuda_worker.zig:189`) |
| 14-…no-tp-graphs.second | 2.28.9 | — | new, empty | not serving after 11 min | cost timing, width 23 (main in `cuStreamSynchronize`) | pair, from `ncclRecv` (`cuda_worker.zig:171`) |
| 15-1.1.0-source.cost-tables-present | 2.28.9 | — | cost tables from a one-GPU start of the same binary | serving after 24 s; request 1 done (400 tokens), request 2 never finished (20-min limit) | — | — |
| 16-1.1.0-debug.nccl-2.32.3 | 2.32.3 | — | new, empty | serving after 69 s; 17 requests done, then our client time cap | idle at stop | idle at stop (`cuda_worker.zig:172`) |
| 17-1.1.0-release.nccl-2.32.3 | 2.32.3 | — | new, empty | serving after 61 s; 18 requests done, client exit 0 | — | — |
| 18-, 19-1.0.12-release.earlier-session-a/b | 2.28.9 | — | rank 0's own | served; hung after 2 and 6 requests (no stacks, rank 0 log only) | — | — |

The client in runs 15-17 is `tools/bench.py decode count prose code` from
https://github.com/plotarmordev/Nemotron-3.5-Super-VL-Single-DGX-Spark-TensorFold (f10fdadaf244bf432427e55d183a317627e7ad04):
greedy, thinking off, each prompt drafted and with `"draft": false`. Its tok/s in these runs include thermal waits
(`cooling:` lines) and are not speed figures.
