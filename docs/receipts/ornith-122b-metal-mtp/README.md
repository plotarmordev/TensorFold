# Ornith-1.5-122B on one M5 Ultra: plain, DFlash2 and MTP receipts (12 November 2026)

The [recipe](../../recipes/ornith-1.5-122b-metal.md)'s speed and quality figures for its build `630390e6e`. Every
figure it cites is recomputed from these files by `python3 check.py`, which needs only the standard library.

| File | What it holds |
| --- | --- |
| `bench-{plain,dflash,mtp}-p{1,4}-fresh.json` | `tools/bench_concurrent.py` reports, one fresh server per mode and capacity, with the recipe's flags and `--levels 1` (p1) or `--levels 1,4 --alone` (p4; drafted modes add `--serial`) |
| `bench-*-fresh.raw.json` | The same runs' per-request records in order (warmup, equality controls, then the timed replies): TTFT, decode rate, token SHA-256 and the reply's `speculative` counters, plus `uptime` before each group |
| `bench-mtp-p1-restart-{0,1,2}.json`, `.raw.json` | One 256-token MTP request on a `--parallel 1` server restarted before each repetition |
| `serve-*.log`, `health-*-fresh.json` | Those servers' logs and `/health` after each run |
| `quality-plain.json`, `quality-dflash.json` | Each mode's greedy 128-token ids for the 16 prompts of `tools/bench_prompts.json`, sent one at a time on a `--parallel 1` server, beside mlx-lm's saved ids (`reference_ids`, from [`mlx_ref.py`](../ornith-122b-metal/mlx_ref.py)) and the leading match |
| `quality-mtp-shared.json` | The same prompts on a `--parallel 4` MTP server in four concurrent groups of four, with each reply's `speculative` counters |
| `score_saved.py`, `score-nll.json` | Mean per-token NLL of `quality-plain.json`'s native and mlx-lm ids under mlx-lm 0.32.2 (MLX 0.32.4), one row per prompt |

Every server ran `TF_QWEN27_SLOT_RESUME=0 tensorfold-native serve ~/models/Ornith-1.5-122B-MLX-4bit --name bench
--context 32768 --prompt-cache-gib 0 --no-thinking` with `--parallel 1` or `4` and one of `--no-drafts`,
`--drafter ~/models/Ornith-1.5-122B-DFlash2` or `--mtp-drafts 1 --drafter ~/models/Ornith-1.5-122B-mtp`.
The raw request records came from a local wrapper around `tools/bench_concurrent.py` that kept each reply's
`speculative` object; it is not committed, and `check.py` reads only what it wrote.

Scoring, with no TensorFold server running so one model holds the GPU:

```sh
python3 score_saved.py ~/models/Ornith-1.5-122B-MLX-4bit quality-plain.json score-nll.json
```
