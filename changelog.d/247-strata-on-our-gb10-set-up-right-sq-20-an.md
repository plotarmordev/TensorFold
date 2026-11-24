# Strata Inkling 2 Mini HTTP bench

- `strata-job.sh`'s `[http]` cells report the end-to-end rate on the client's clock (usage tokens over the makespan, prefill included, median of 3) and fail only on a failed request or an untimed rep. Strata v0.2.19's server drops the `<|content_thinking|>` span Inkling opens even with thinking off, so its 128-token replies can stream no text and decode has no token arrivals to time; each line gives the completion classes that show it.
- A failed `[http]` cell prints the server's `[strata]` lines (run with `STRATA_DEBUG=1`).
- `strata-http-sq20.sh` and `strata-http-sq10.sh` run one pack's HTTP bench at 1, 4 and 64 streams without repeating the own bench (`STRATA_OWN=0`).
