# Strata Inkling 2 Mini HTTP bench diagnosis

- An unmeasured or failed `[http]` cell in `strata-job.sh` now names its cause per request (completion class, finish reason, usage tokens, text pieces and their arrival times) and prints the server's `[strata]` lines (run with `STRATA_DEBUG=1`), instead of a bare `0.0 tok/s`.
- Every `[http]` line also gives the end-to-end rate (usage tokens over the makespan, prefill included), labelled apart from the decode rate.
- `strata-http-sq20.sh` and `strata-http-sq10.sh` run one pack's HTTP bench at 1, 4 and 64 streams without repeating the own bench (`STRATA_OWN=0`).
