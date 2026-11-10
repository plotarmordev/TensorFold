# Ornith-1.5-122B on one M5 Ultra: receipts

Benchmark and quality receipts were taken on 9 November 2026 on a Mac17,15 (Apple M5 Ultra, 256 GiB) with
`tensorfold-native 1.2.4` built from `ffab654aa` (upstream main `5aca5c051` merged). Health snapshots and hash
verification were repeated on fresh servers and the local downloads on 10 November. The
[recipe](../../recipes/ornith-1.5-122b-metal.md) gives the run commands and reads these files.

| File | What it holds |
| --- | --- |
| `hashes.txt` | Every LFS file's size and expected/actual SHA-256 from Hugging Face metadata and the local download at the pinned revisions |
| `health.txt` | `/health` idle and peak memory snapshots for plain and DFlash2 serving on fresh `--parallel 1` and `--parallel 4` servers |
| `bench-plain.json`, `bench-dflash.json` | `tools/bench_concurrent.py --levels 1,4 --tokens 256 --reps 3 --temperatures 0 --prompts tools/bench_prompts.json --alone` on a `--parallel 4` server, `--no-drafts` and `--drafter`; the drafted run adds `--serial` |
| `bench-plain-p1.json`, `bench-dflash-p1.json` | The same prompts and settings at one stream on fresh `--parallel 1` servers, run without `--alone` |
| `mlx_ref.py` | mlx-lm 0.32.2 (MLX 0.32.4) greedy ids for `tools/bench_prompts.json`, 128 tokens, chat items through the checkpoint's template with thinking off |
| `tf_match.py`, `match-plain.txt` | TensorFold's greedy ids for the same prompt ids, leading and positional matches against mlx-lm's |
| `split_gap.py`, `split-gap.txt` | At each prompt's first split, mlx-lm's log-probability of its own token minus TensorFold's, from a full forward over the shared prefix |
| `score_nll.py`, `score-nll.txt` | Mean per-token NLL of both continuations under mlx-lm's teacher-forced forward of the publisher's weights |

The scripts take the server URL, the checkpoint folder and the reference file as arguments, for example
`python3 score_nll.py http://127.0.0.1:8902 ~/models/Ornith-1.5-122B-MLX-4bit ref-128.json 128`.
