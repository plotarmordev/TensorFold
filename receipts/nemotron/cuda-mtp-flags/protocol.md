# Qualification protocol

Candidate: `0a0b2ca5173075ec0bfd96a58762dd4f52dc7f8c`; base: `6ee828c24bf802cdd73507d6fc60bb27a82c0d66`.

`qualify.py LABEL BINARY COMMIT [FLAGS...]` runs inside the GPU lock with a `timeout --signal=TERM 19m` deadline. The runner sets the thermal guard, monitors temperature and memory, sends SIGTERM on cleanup, and records whether the GPU has no remaining compute processes before returning. Each completed JSON records the checkpoint, driver, CUDA compiler and serve flags. Private home paths and the loopback address in server logs are replaced with placeholders.

The default sequence uses the same DGX Spark: `base-a`, `pr-a`, `pr-b`, `base-b`. Flag checks run on a second DGX Spark: `cap0` uses `--mtp-drafts 0`; `cap1` uses `--mtp-drafts 1 --mtp-confidence 0`; `confidence1` uses `--mtp-confidence 1`.

Each hold first compares greedy code and prose replies against `draft: false`, recording token hashes and draft/accepted/round counter deltas. It then invokes the upstream clients:

```
python3 tools/bench_concurrent.py <endpoint> super --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --label <label> --output <concurrent.json>
python3 tools/bench_openai.py <endpoint> super --tokens 256 --reps 3 --temperatures 0 --label <label> --output <decode.json>
```

Use the benchmark clients from the candidate tree beside `qualify.py` in `tools/`. `compare.py <receipt-directory>` checks every captured candidate and base reply against the first base's plain hashes, checks the clients' solo/concurrent/plain results, and reports decode medians.

The draft counter includes copied continuations. A nonzero MTP cap does not cap copied chains; the code prompt's total drafts can therefore exceed its rounds even with `--mtp-drafts 1`. The comparison checks the observed reduction from default rather than claiming this combined counter is exclusively MTP.

An initial launch attempt could not find `nvcc` on the remote PATH and stopped before starting a server. The runner now uses the installed compiler's absolute path. This attempt is not counted as a model test.
