# Ornith-1.5-122B on one M5 Ultra: plain, DFlash2 and MTP receipts (13 November 2026)

The [recipe](../../recipes/ornith-1.5-122b-metal.md)'s speed and quality figures. Every figure it cites is recomputed
from these files by `python3 check.py`, which needs only the standard library.

| File | What it holds |
| --- | --- |
| `bench-{plain,dflash,mtp}-p{1,4}-fresh.json` | `tools/bench_concurrent.py` reports from build `9f7f42296`, one fresh server per mode and capacity, with the recipe's flags and `--levels 1` (p1) or `--levels 1,4 --alone` (p4; drafted modes add `--serial`) |
| `bench-*.raw.json` | The same runs' per-request records in order (warmup, equality controls, then the timed replies): TTFT, decode rate, token SHA-256 and the reply's `speculative` counters, plus `uptime` before each request |
| `health-*-fresh.json` | Those servers' `/health` after each run |
| `quality-plain.json`, `quality-dflash.json` | 12 November (build `630390e6e`): each mode's greedy 128-token ids for the 16 prompts of `tools/bench_prompts.json`, sent one at a time on a `--parallel 1` server, beside mlx-lm's saved ids (`reference_ids`, from [`mlx_ref.py`](../ornith-122b-metal/mlx_ref.py)) and the leading match |
| `quality-dflash-p1.json`, `quality-mtp-p{1,4}.json` | 13 November (build `ed18f8a08`): replies to the same prompts, one at a time (p1) or in four concurrent groups of four (p4): each reply's server token SHA-256, whether it equals `quality-plain.json`'s, and its `speculative` counters |
| `score_saved.py`, `score-nll.json` | Mean per-token NLL of `quality-plain.json`'s native and mlx-lm ids under mlx-lm 0.32.2 (MLX 0.32.4), one row per prompt |

Every server ran `TF_QWEN27_SLOT_RESUME=0 tensorfold-native serve ~/models/Ornith-1.5-122B-MLX-4bit --name bench
--context 32768 --prompt-cache-gib 0 --no-thinking` with `--parallel 1` or `4` and one of `--no-drafts`,
`--drafter ~/models/Ornith-1.5-122B-DFlash2` or `--mtp-drafts 1 --drafter ~/models/Ornith-1.5-122B-mtp`.
The raw request records came from a local wrapper around `tools/bench_concurrent.py` that kept each reply's
`speculative` object and waited before each request while the one-minute load was above 36; it is not committed, and
`check.py` reads only what it wrote. The first plain one-stream server measured 62.0 tok/s while the drafted
four-capacity servers' `draft:false` controls ran at 85.0–92.1; `bench-plain-p1-fresh` is its repeat on a fresh
server. The MTP one-stream server ran in the same minutes as that disturbed run and was not repeated before the GPU
window closed. A shared round's four replies carry that round's `speculative` counters; `check.py` counts each
four-stream group once.

`quality-*-p{1,4}.json` came from build `ed18f8a08`, which differs from `9f7f42296` only in how a lone MTP stream
prices its drafts (how many it proposes, not which tokens it keeps). On `9f7f42296`, every timed `quicksort-raw` reply
in every mode has the same token SHA-256 (`check.py` asserts it) and every four-stream reply equals its solo reply.

Scoring, with no TensorFold server running so one model holds the GPU:

```sh
python3 score_saved.py ~/models/Ornith-1.5-122B-MLX-4bit quality-plain.json score-nll.json
```
