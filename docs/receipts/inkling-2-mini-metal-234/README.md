# Inkling 2 Mini on Metal, M5 Ultra: card 234's receipt

Apple M5 Ultra, 256 GB, macOS 27.0, the official checkpoint with its own `mtp.safetensors` as the DSpark drafter,
`tensorfold-native` from `zig build native -Doptimize=ReleaseFast`, `--context 65536`. The Desk's Mac GPU window,
24 Nov 02:02-02:38 HKT, one lock held for the whole sequence (`run.sh`); the load average at each step is in its line
below, since other cards' CPU work kept running.

Control: card 210's tip `781eeab52`. Candidate: `630378dc0` on that base (a verify window keeps its convolution
histories in place; every stream's keep shares one command buffer).

## Ceiling

A plain step reads about 5.6 GB: 8 of 128 experts (5.1 MB each) in 48 layer passes (24 layers, the 12 core ones three
times) is 1.96 GB, the other layer weights 2.7 GB over the same passes, the bf16 LM head 0.93 GB. Card 210's 16.1 ms
plain step is 350 GB/s; the M5 Ultra's bandwidth was not measured in this window, so the percentage of ceiling is not
claimed. DSpark's bar is the hold: a round costs 1.74 plain steps at one stream and 2.24 at four.

## DSpark hold lever

`bench.sh` (card 210's): medians of three reps, `--levels 1,4 --tokens 256 --temperatures 0 --alone --serial`.

| tok/s (code / chat) | control | candidate | control | candidate | #4618 |
|---|---|---|---|---|---|
| 1 stream | 118.9 / 96.8 | 120.2 / 93.3 | 122.2 / 45.6* | 118.0 / 95.3 | 123 |
| 4 streams | 275.7 / 227.7 | **282.7 / 233.6** | 274.9 / 190.3* | **285.7 / 235.7** | 281 |
| 4-stream round, code | 39.3 ms | 38.5 ms | 39.4 ms | 37.9 ms | |

\* reps 81/32/46 and 112/190/216: disturbed, not used.

- At four streams the lever is kept: code beats both controls by 2.5-3.9% and #4618's 281 by 0.6-1.7%, chat beats the
  clean control by 2.6-3.5%, a round is 1 ms shorter at the same 2.81 tokens.
- At one stream it is inside noise (118.0-120.2 against 118.9-122.2) and under #4618's 123: not beaten. It passes
  Strata v0.2.19's 108.9-114.5 on M5 Max on code (another chip: a deployment comparison).
- Plain, candidate: 61.1 / 62.6 at one stream, 223.7 / 224.1 at four, as card 210's 61.0-62.3 / 216-221. The plain
  control beside it was disturbed (reps 188/173/171, 50/51/51) and is not used.
- Every cell: 0 failed requests, every concurrent reply equal to its run alone (12 of 12), drafted equal to plain.

## Prefill at 45K

`prefill.sh` with the drafter on: **2,365 tok/s** median (2,164-2,368), first token in 19.5-21.3 s; card 210 read
2,439 without the drafter. Each repeat equals its fresh reply but reads 0 cached tokens with `--drafter`, where card
210's plain session read 46,080: the drafter session does not reuse the prompt cache.

## Engines at 1 and 64 streams: void

Card 235's `engines.sh` ran next (02:12-02:38), when the load average rose from 35 to 72: TensorFold's own one-stream
code cell fell to 100.2 (99-117) from 118-120 minutes earlier, and oMLX's code cells streamed each reply in one chunk
after 13 s, so they are unmeasured. No engine comparison is claimed from it; card 235 owns that receipt.

## Files

`cells.jsonl`: every cell (medians, reps, round, `alone`, `serial`); `prefill45k.txt`; `run.sh` reproduces the
sequence.
