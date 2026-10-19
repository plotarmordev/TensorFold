# Data for the shared-round draft-depth issue

All runs: Nemotron 3.5 Super VL NVFP4 (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`)
on a DGX Spark (GB10, driver 580.173.02, CUDA 13.0); the nsys pair on a second DGX Spark. Every server is a fresh
process with an empty window-cost cache, so it times its own table at start, served as

    TF_HEAT_HIGH=85 TF_HEAT_LOW=75 tensorfold-native serve <checkpoint> --host 127.0.0.1 --port 8897 \
        --name nemotron-3.5-super-vl --context 131072 --parallel auto --vision --thinking

## Builds

Clean checkouts, `-Doptimize=fast -Dstrip=true -Dsm=121 -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline`.

| Label | Source | `tensorfold-native` sha256 |
|---|---|---|
| `1.1.0` | v1.1.0, 76aafcf274e60a5e31a59ce145b2f71f7b366e9d | `04167a172c847952737a4b401fb5945b74d598a51312ed19f4e6ddd974015088` |
| `1.1.0+timing-1.0.12` | v1.1.0 + `timing-1.0.12.diff` (9a6bc0c6 reverted, one pool over every row as before 480ad61f) | `270ca9e9e5e7516941f615ef7f1ae9948a8cf39a1350d528c5e85f163dca8b16` |
| `1.1.0+explore` | v1.1.0 + `explore.patch` (the first unverified depth priced as the one before it) | `b302a697ef31217dd6dd804902698283dc983c59fb0b80539bbceecc7ce79aae` |
| `1.1.0+overlap+timing-1.0.12` | v1.1.0 + the shared-expert overlap change + `timing-1.0.12.diff` (`c1.txt` only; the overlap changes shared rounds only) | `0fb6df7cf8a0b9d22d13e2f0ba47c67fbc316ad054976c8e5fa672ddca06106f` |

## Client

Chat completions, greedy (`"temperature": 0`), `"ignore_eos": true`, `"max_tokens": 400`,
`"chat_template_kwargs": {"enable_thinking": false}`, streamed. Prompts:

- count: `Count from 1 to 1000. Output every number, separated by single spaces. Do not abbreviate or use ellipses. No other text.`
- prose: `Write a quiet short story about a lighthouse keeper who receives a letter addressed to someone who left the island forty years ago.`
- code: `Write a Python function that parses an ISO 8601 duration such as P3DT4H5M6S into total seconds, with type hints, a docstring and three unit tests.`
- count-distinct: stream k sends `Stream k: ` before the count prompt, so no two streams share tokens.

Per case: the reply alone first (one per stream for distinct cases), then a warmup wave and 3 measured waves of N
requests released together. Before each wave, a box hotter than 85 °C is let cool to 75 °C.

## Files

- `waves.jsonl`: one line per wave. `set` (`bisect`, `distinct`, `explore`, `c2`: one server series each), `hold`
  (run order within the set), `build`, `case`, `n`, `wave` / `warmup`, `agg_tps` (output tokens over dispatch to
  the last token), `decode_tps` (first text to last), `first_ms` (each request's first text after dispatch, sorted),
  `server_ttft_ms`, `equal_alone` (replies equal to their reply alone), `late`: out of step, i.e. request 2's first
  text trails request 1's by more than 1.5 times the gap from request 2 to request 3 (never set for n = 2).
- `summary.md`: per hold and case, the median of the 3 measured waves, each wave (`*` = out of step), rounds and
  accepted/drafted a request from the server's `done` lines.
- `done.txt`: the server's `done` line for every request, in order, prefixed by set, hold and build.
- `c1.txt`: one stream at a time (warmup + 3 measured, median of the server's decode tok/s; 400 tokens count, 256
  prose and code), drafted and `"draft": false` token hashes, and each request's rounds.
- `tables/*.bin`: each server's whole-head window-cost table, the one shared rounds are priced with
  (`cuda_lanes.zig` `measure`). Layout: `TFWC2\n`, 33 little-endian f64 (verify ms by rows 0..32), u64 rows, f64
  head level ms.
- `sim.py`: replays `Rule.budgets`, the trim in `windows.allocate` and `observeDepth` on a table.
  `bars.py`: the least acceptance at depth k for which budgets grants a k-th draft at c4. `sim.txt`: both, as run.
- `timing-1.0.12.diff`, `explore.patch`: the two changes above, against v1.1.0.
- `nsys/*_cuda_gpu_kern_sum.csv`: `nsys stats` kernel summaries of two count c4 waves on `1.1.0`, one in step
  (released together, 72/72/72/72 rounds) and one out of step (requests 2-4 sent 150 ms after request 1,
  70/72/72/72 rounds); all 8 replies equal.
