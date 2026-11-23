# Inkling 2 Mini on Metal, M5 Ultra: card 234's receipt

Apple M5 Ultra, 256 GB, macOS 27.0, the official checkpoint with its own `mtp.safetensors` as the DSpark drafter,
`tensorfold-native` from `zig build native -Doptimize=ReleaseFast`, `--context 65536`. The Desk's Mac GPU window,
24 Nov 02:02-02:38 HKT, one lock held for the whole sequence (`run.sh`); the load average at each step is in its line
below, since other cards' CPU work kept running.

Control: card 210's tip `781eeab52`. Candidate: `630378dc0` on that base (a verify window keeps its convolution
histories in place; every stream's keep shares one command buffer).

## Ceiling

A plain step reads about 5.6 GB: 8 of 128 experts (5.1 MB each) in 48 layer passes (24 layers, the 12 core ones three
times) is 1.96 GB, the other layer weights 2.7 GB over the same passes, the bf16 LM head 0.93 GB. `bandwidth.py` over
4 GiB arrays, six timed passes each, at 05:04 (load 123, the GPU otherwise idle):

| pass | GB/s (median) | ms a pass |
|---|---|---|
| read (sum) | 1,021 | 4.20 4.14 4.15 5.59 4.22 4.26 |
| add (2R+1W) | 1,029 | 13.18 12.89 12.52 12.53 12.45 12.36 |
| copy (1R+1W) | 1,004 | 8.48 8.55 8.53 8.59 8.56 8.56 |

Three earlier runs at 03:46, beside other cards' GPU work, read 388-991 and are not used. At 1,020 GB/s a plain step's
floor is 5.5 ms, 182 tok/s, and at a practical 75% of it 137 tok/s. The 16.1 ms plain step (62 tok/s) is 34% of the
floor; DSpark at one stream, 3.33 tokens a 28.2 ms round, is 20% of its 3.33 × 182 bound. Bytes do not set the step:
its dependent chain of kernels does, so the hold lever cuts launches and waits, not reads.

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
- Plain, 05:00-05:04, control / candidate / control with the load average at 80, 90 and 127:

  | plain, tok/s (code / chat) | control | candidate | control |
  |---|---|---|---|
  | 1 stream | 51.9 / 55.5 | 52.0 / 60.1 | 60.4 / 50.4 |
  | 4 streams | 205.0 / 169.7 | 159.0 / 179.1 | 204.9 / 155.7 |
  | round p50, 1 / 4 streams, code | 16.9 / 18.2 ms | 16.7 / 18.3 ms | 16.4 / 18.1 ms |

  Host load moved single reps by up to 40% (control code at four streams: 205.0, 127.2, 209.5), so the medians cannot
  separate the builds; the per-round time, which the load does not reach, can: it is equal within 0.5 ms at both
  stream counts, as the lever only touches the drafted path. The earlier candidate cells (61.1 / 62.6 and 223.7 / 224.1
  at load 9) match card 210's 61.0-62.3 / 216-221.
- Every cell: 0 failed requests, every concurrent reply equal to its run alone (12 of 12), drafted equal to plain.

## Prefill at 45K

`prefill.sh` with the drafter on: **2,365 tok/s** median (2,164-2,368), first token in 19.5-21.3 s; card 210 read
2,439 without the drafter. Each repeat equals its fresh reply but reads 0 cached tokens with `--drafter`, where card
210's plain session read 46,080: the drafter session does not reuse the prompt cache.

## Engines at 1 and 64 streams

oMLX's own 4-bit MTP build `Inkling-2-Mini-oQ4e-mtp`, the fastest other engine on this Mac, from card 235's
`engines.sh`; TensorFold from `engines.sh` here, DSpark on 64 lanes. Card 235's script started TensorFold on its
default 8 lanes, so at 64 streams 56 requests queued (168 of 192 waited 10-55 s) and read 141.4 / 154.8; that is fixed
here, and no request queued. All runs were under other cards' CPU load (35-77). oMLX streams each code reply in one
chunk after 13 s, so its code cells are unmeasured in every run, card 235's too; the comparison is on chat.

| chat, tok/s total | TensorFold, 64 lanes (03:47-03:53) | oMLX (02:25 / 03:00) |
|---|---|---|
| 1 stream | **90.2** (89.5-91.5) | 38.4 / 24.8 |
| 64 streams | **306.6** (229.4-311.2) | 225.6 / 203.3 |

- TensorFold leads at both: 2.3× at one stream, 1.36-1.51× at 64.
- TensorFold's code cells in the same session: 115.4 at one stream (96.5-117.7), 239.8 at four (194.3-273.5, load
  60-77, under the clean 282.7-285.7 above), 399.1 at 64 (309.3-456.4).
- 0 failed requests in every cell of both engines.

## Files

`cells.jsonl`: every cell (medians, reps, round, `alone`, `serial`); `prefill45k.txt`; `bandwidth.py` measures the
ceiling's bandwidth; `engines.sh` runs TensorFold's receipt cells; `run.sh` reproduces the sequence.
