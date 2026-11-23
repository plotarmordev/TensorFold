# Inkling 2 Mini on Metal, M5 Ultra: card 210's receipt

Apple M5 Ultra, 256 GB, macOS 27.0; the official checkpoint (`Inkling-2-Mini`, its own `mtp.safetensors` as the
DSpark drafter); `tensorfold-native` from `zig build native -Doptimize=ReleaseFast` at 930686a0, `--context 65536`.
Every run held `/tmp/tensorfold-model.lock` inside the Desk's Mac GPU window, one at a time, nothing else on the GPU.

## Speed

`bench.sh` runs `tools/bench_concurrent.py --levels 1,4 --reps 3 --tokens 256 --temperatures 0 --alone --serial`:
medians of three reps, the default code and chat prompts, greedy. Control, candidate, control:

| tok/s (code / chat) | plain, first | DSpark | plain, again | best open number |
|---|---|---|---|---|
| 1 stream | 61.0 / 62.0 | **120.7 / 97.8** | 62.2 / 62.3 | #4618: 123 (DSpark) |
| 4 streams, total | 220.8 / 219.2 | **279.6 / 226.2** | 219.6 / 216.2 | #4618: 281 (DSpark) |

- 0 failed requests in every cell; every concurrent reply equal to its run alone (12 of 12 at 4 streams, 3 of 3
  alone) and the drafted replies equal to the plain ones (`serial`).
- DSpark on code: 3.33 tokens a round at one stream (28.2 ms a round against 16.2 ms a plain step), 2.81 at four
  (39.0 ms against 17.4 ms). The two plain controls agree within 2%.
- Against the bar: not beaten. One stream with DSpark is 2% under #4618's 123 and four streams 0.5% under its 281;
  the plain controls moved 2% between them, so this is a tie at best. Strata v0.2.19's 108.9-114.5 on M5 Max is
  passed at one stream on code (another chip: a deployment comparison, not a chip one). The gap is the hold, not
  acceptance: a round costs 1.74 plain steps at one stream and 2.24 at four, for 3.33 and 2.81 tokens.

## Prefill at 45K

`prefill.sh` sends three distinct prompts of 46,155-46,221 tokens cold, then each again: **2,439 tok/s** median
(2,430-2,441), first token in 18.9-19.0 s. The repeat reads 46,080 cached tokens, answers in 0.14-0.16 s and its
reply is identical to the fresh one.

## Exactness

With `TF_INKLING2_METAL_DIR` and `TF_INKLING2_DSPARK_DIR` set to the checkpoint, the `test-inkling2-metal-host` test
binary passes all 85 of its tests on the device (175 s), among them:

- `Mini Metal serving matches the host reference token for token`: the first turn, a fresh second turn and the
  same turn restored from the prompt cache each equal the host reference, logprobs included;
- `Mini Metal opt-in DSpark drafting preserves target tokens and rolls back exactly`.

`TF_INKLING2_METAL_DIR=$HOME/models/Inkling-2-Mini zig build test-inkling2-metal-float64` passes the official-checkpoint
Metal gate: 0 of 32 greedy tokens differ from the teacher-forced float64 host reference, while the planted control
without cross-pass messages disagrees. The host reference follows the shipped Metal expert activation's bf16 rounding
of gate and up before softcapped SwiGLU; the other arithmetic stays float64. The test holds
`/tmp/tensorfold-model.lock` only while the Metal model runs, then closes the model before the two host passes; the
full gate took 13 minutes (mostly CPU). The Debug host suite skips this slow test; the ReleaseFast step runs it.

## Files

`cells.jsonl`: each cell of the three bench runs (bench_concurrent's medians, its `alone` and `serial` counts; the
full outputs carry token hashes and stay in the run's `OUT`), `prefill45k.txt`; `bench.sh`, `prefill.sh` and
`prefill.py` reproduce them (`MODEL` names the checkpoint, `OUT` the scratch folder).
