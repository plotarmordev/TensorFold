# Follow-up receipts: nemotron/shared-expert-overlap (#1002)

Same binaries as the published receipt, no code change: base `ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba`
(`tensorfold-native` sha256 `f24658bd2f308a05495928e18f9d9e8240323bc11f4573feaef35deaefbdb93f`) and tip
`14c78e735cc9effb2dfb2c2dc53c54b93d06f481` (`33d9dcf6e0064903ce08f723fd6b3559f8b3f03828e71200c9c02e5a314e8d6f`), both
from clean checkouts (`../builds.txt`). A DGX Spark (GB10) for `calibration/` and `table-swap/` (the node of the
published holds), a second for `resume/`; driver 580.173.02; `TF_HEAT_HIGH=85 TF_HEAT_LOW=75`. Every hold's `.json`
and every client JSON carries `source_commit` and `binary_sha256`; each hold `.json` also lists the window-cost files
under the server's `HOME` before and after (name and sha256).

## Window-cost tables (`tables/`, `calibration/`)

A CUDA server times its decode windows at its first start and keeps the table at
`$HOME/.cache/tensorfold/window-costs/<key>.bin`, the key a hash of the executable's bytes and the GPU and model shape
(`core/lanes/cost_cache.zig`, `families/nemotron/cuda_costs.zig`); later starts of the same binary read it back.
`calibration/cal-{base,tip}`: each build started once with an empty `HOME` of its own; the file it wrote gives its
key: base `249d4493496321b6d191c06384c9191b59531941094a9a01c9b52f2609dcb6ea.bin`, tip
`c620f43fd763e4de5859f1f33b55f60f91a75e2bfb788778f49e677660b39478.bin` (both also wrote the shared reference file
`0ab3edd7….bin`). The published holds read the files under those two names in the measurement node's own cache, written
by each build's first published hold (base 22:34:16, tip 22:47:25) and copied here as `tables/*-published.*.bin`;
`tables/*-recalibrated.*.bin` are the calibration starts' tables. `tables/tables.md` decodes all four.

## Same table, both builds (`table-swap/`)

`homes/with-base-table` holds the base's published table under both keys, `homes/with-tip-table` the tip's under both
(`scripts/swap.sh`). Eight fresh servers, in this order: base, tip, tip, base reading the base's table, then tip, base,
base, tip reading the tip's. Each runs `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4
--alone --serial` with `serve --name super --no-update-check --prompt-cache-gib 0`; no hold wrote a table (before ==
after in every `.json`). `table-swap/summary.md` has every cell; the round counts come from the server logs' `done`
lines.

## Resumed against fresh (`resume/`)

Default serve flags (16 GiB prompt cache), tip only. `turns-1`: the 8,186-token chat prompt's messages (thinking off,
greedy, 256 tokens, `ignore_eos`), then a second turn (those messages, the first reply as the assistant turn, a
follow-up question; 8,467 prompt tokens), the second turn again with `"draft": false`, and the first turn again.
`fresh-1`: a new server, the same second turn cold. The first reply's text is not published (sha256
`04f6a09ef418b3028fdb933a16fae1b87f872d795da60f096c90661d7b2e192d`). Raw-token `/v1/completions` requests carry no
chat history and keep no prompt state, so the check uses chat turns.

`scripts/`: `hold.py` (one hold), `calibrate.sh`, `swap.sh`, `resume.sh`; `<gpu.lock>` and `../builds/` stand for the
node's lock file and the directory of the two binaries. The calibration holds ran an earlier revision of `hold.py` that
differed only in the request modes (its `start` mode sends no request).
