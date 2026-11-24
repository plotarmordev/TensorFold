# Inkling 2 Mini on Metal, M5 Ultra: card 210's receipt

Apple M5 Ultra, 256 GB, macOS 27.0; the official checkpoint (`Inkling-2-Mini`, its own `mtp.safetensors` as the
DSpark drafter); `tensorfold-native` from `zig build native -Doptimize=ReleaseFast`, `--context 65536`: the speed
table at 63aacbfc (code as at d29431d5f), the 45K prefill at 930686a0. Every run held `/tmp/tensorfold-model.lock`
inside the Desk's Mac GPU window, one at a time.

## Ceiling

A device-to-device blit of 4 GiB on this M5 Ultra moves **1,072 GB/s** (read plus write, best of 12; `blit.swift`
repeats the measurement and `blit.txt` holds its output: a re-run came out at 1,035, 3.5% under it). Bytes a plain
step reads, from the checkpoint's safetensors headers, over the 48 visits (layers 6-17 three times): 2,787 MB of
dense layer weights, 1,839 MB of routed experts (8 of 128 a visit), 926 MB of the bf16 unembedding: 5.55 GB a token.
At four streams a step reads the dense weights and the head once and 22.8% of each visit's experts (1 - (120/128)^4):
10.41 GB for four tokens. DSpark adds the drafter's 636 MB a round and divides the target's bytes by the accepted
length. KV at 256 tokens is under 0.1% and left out.

| ceiling, tok/s | plain | DSpark (code, measured accepted length) |
|---|---|---|
| 1 stream | 193 | 3.303 tokens a round: 572 |
| 4 streams, total | 412 | 2.813 tokens a round: 1,092 |

The measured cells below are 32% (plain) and 21% (DSpark) of their ceiling at one stream, 54% and 26% at four: the
step is bound by the 48 visits' launch chain, not by bandwidth, and a DSpark round is not yet priced like a plain step.

## Speed

`bench.sh` runs `tools/bench_concurrent.py --levels 1,4 --reps 6 --tokens 256 --temperatures 0 --alone --serial`:
medians of six reps, the default code and chat prompts, greedy. Control, candidate, control (24 Nov, 08:23-08:46):

| tok/s (code / chat) | plain, first | DSpark | plain, again | best open number |
|---|---|---|---|---|
| 1 stream | 62.3 / 63.3 | **122.5 / 91.9** | 61.5 / 62.2 | #4618: 123 (DSpark) |
| 4 streams, total | 223.3 / 224.8 | **279.6 / 228.0** | 219.8 / 221.0 | #4618: 281 (DSpark) |

- 0 failed requests in every cell; every concurrent reply equal to its run alone (24 of 24 at 4 streams, 6 of 6
  alone) and the drafted replies equal to the plain ones (`serial`). The two plain controls agree within 1.8%, p95
  round gaps 16.3-18.0 ms in both.
- DSpark on code: 3.30 tokens a round at one stream (27.1 ms a round against 15.9 ms a plain step), 2.81 at four
  (39.0 ms against 17.2 ms).
- Against the bar: #4465's 97 and 184-223 are passed on code; #4618's 123 / 281 are not beaten (0.4% and 0.5% under,
  inside the controls' 1.8% drift, a tie at best). Strata v0.2.19's 108.9-114.5 on M5 Max is passed at one stream
  on code (another chip: a deployment comparison, not a chip one). On chat, DSpark accepts 2.29 tokens a round and
  one stream gives 91.9. The gap is the hold, not acceptance: a round costs 1.70 plain steps at one stream and 2.26
  at four.
- Void and left out: the second control twice ran beside another card's `zig build test` (Metal kernel tests,
  outside the lock): p50 gaps held at 16-19 ms but p95 rose to 55-84 ms and the totals fell 10-42%. Those runs'
  per-cell figures were not kept, so `cells.jsonl` carries only the measured runs. It was rerun once no test
  process was left. On code, the earlier three-rep table (120.7 / 279.6) agrees with this one within 1.5%.

## Prefill at 45K

`prefill.sh` sends three distinct prompts of 46,155-46,221 tokens cold, then each again: **2,439 tok/s** median
(2,430-2,441), first token in 18.9-19.0 s. The repeat reads 46,080 cached tokens, answers in 0.14-0.16 s and its
reply is identical to the fresh one.

## Exactness

With `TF_INKLING2_METAL_DIR` and `TF_INKLING2_DSPARK_DIR` set to the checkpoint, the `test-inkling2-metal-host` test
binary at `d29431d5f` passed 85 device tests and skipped one, the float64 gate it skips in Debug (383 s, under the
lock). These include:

- `Mini Metal serving, fresh and restored, matches one serial stream on the backend token for token`: the first
  turn, a fresh second turn and the same turn restored from the prompt cache each equal one stream run alone on the
  same Metal backend, logprobs included. It checks serving against the backend, not against the host: the host
  oracle is the float64 gate below;
- `Mini Metal opt-in DSpark drafting preserves target tokens and rolls back exactly`.

`TF_INKLING2_METAL_DIR=$HOME/models/Inkling-2-Mini zig build test-inkling2-metal-float64` passes the official-checkpoint
Metal gate: 0 of 32 greedy tokens differ from the teacher-forced float64 host reference, while the planted control
without cross-pass messages disagrees. The host reference follows the shipped Metal expert activation's bf16 rounding
of gate and up before softcapped SwiGLU; the other arithmetic stays float64. The test holds
`/tmp/tensorfold-model.lock` only while the Metal model runs, then closes the model before the two host passes; the
full gate took 13 minutes (mostly CPU). The Debug host suite skips this slow test; the ReleaseFast step runs it.

## CUDA compile records

The TOP `docs/recipes/inkling-2-mini-cuda/limits-job.sh` compile-only run at `0b71ebdb8` printed the real
nvcc register, static shared-memory and thread-bound records on sm_80/86/89/120/121 for the ten
`inkling2_modelopt` and `inkling2_loop_math` launches. The stale-record audit expectedly exited 1 with 40 mismatches;
those `RECORD` lines replaced the zero placeholders in `inkling2_experts_records.zig`. The subsequent local
`zig build test test-golden` passed (1,038/1,038 steps, 9,183 passed, 92 skipped).

At `d29431d5f` the same job's contract step on TOP's GB10 (sm_121) passed `tf_ink2_topk` gates 1-4 with the new
rule, a row out of non-NaN candidates picking -1 at weight 0 (8 shapes exact, five runs byte-identical); its planted
unbiased control went red as planted, and the live limits match the record (36 registers, 384 shared bytes).

## Files

`cells.jsonl`: each cell of the three six-rep bench runs (bench_concurrent's medians, its `alone` and `serial`
counts; the full outputs carry token hashes and stay in the run's `OUT`), `prefill45k.txt`; `bench.sh`, `prefill.sh`
and `prefill.py` reproduce them (`MODEL` names the checkpoint, `OUT` the scratch folder).
