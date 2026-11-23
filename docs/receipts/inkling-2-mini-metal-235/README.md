# Inkling 2 Mini on Metal, M5 Ultra: card 235's receipt

Apple M5 Ultra, 256 GB, macOS 27.0; the official checkpoint (`Inkling-2-Mini`, its own `mtp.safetensors` as the
DSpark drafter); `tensorfold-native` from `zig build native -Doptimize=ReleaseFast`. Control is card 210's head
(781eeab52), candidate this branch; both carry this branch's per-depth draft counters (1de6b1c30), which only report.
Every run held `/tmp/tensorfold-model.lock` inside the Desk's Mac GPU window, one server at a time. Other agents'
builds kept the load average between 50 and 90; the two controls differ by up to 40% (16 chat streams), so the
candidate ran between them.

## Step-major holds

`bench.sh` (`--context 65536 --parallel 16`; `tools/bench_concurrent.py --levels 1,2,4,16 --reps 3 --tokens 256
--temperatures 0 --alone`), medians of three reps, aggregate tok/s. Control, candidate, control:

| code / chat | control | **candidate** | control again |
|---|---|---|---|
| 1 stream | 119.7 / 93.9 | **121.9 / 93.4** | 125.7 / 94.0 |
| 2 streams | 192.9 / 159.2 | **209.8 / 172.8** | 196.0 / 157.6 |
| 4 streams | 278.3 / 228.3 | **317.2 / 259.5** | 277.6 / 225.2 |
| 16 streams | 284.1 / 266.8 | **372.2 / 348.9** | 267.4 / 190.9 |

- Against the better control: +7.0% / +8.5% at two streams, +14.0% / +13.7% at four, +31.0% / +30.8% at sixteen.
  One stream keeps the solo chain (−3.0% / −0.6%, inside the controls' spread).
- 0 failed requests; every concurrent reply equal to its run alone (3, 6, 12 and 48 of each in every session).

Acceptance by depth, head chain rows kept over drafted, summed over each cell's three reps (`drafts.depth_accepted`
and `drafts.depth_drafted` in `cells.jsonl`). From two streams on the three sessions agree row for row; the later
depths' few rows are the first round's:

| code | control | candidate | control again |
|---|---|---|---|
| 1 | 172/232 132/226 103/224 57/171 39/156 25/153 7/78 | 169/229 125/223 94/216 70/216 48/181 29/150 4/45 | 173/228 134/222 105/222 71/201 48/163 7/50 0/4 |
| 2 | 420/534 318/534 252/528 0/6 0/6 | 420/534 318/534 252/528 0/6 0/6 0/6 0/6 | 420/534 318/534 252/528 0/6 0/6 |
| 4 | 840/1068 636/1068 504/1056 0/12 0/12 | 840/1068 636/1068 504/1056 0/12 0/12 0/12 0/12 | 840/1068 636/1068 504/1056 0/12 0/12 |
| 16 | 5424/6720 | 5424/6720 | 5424/6720 |

| chat | control | candidate | control again |
|---|---|---|---|
| 1 | 223/321 113/297 68/261 21/177 10/157 6/58 0/5 | 225/322 112/292 71/267 20/213 9/133 2/15 | 212/307 107/295 77/280 28/219 20/186 13/109 0/10 |
| 2 | 438/648 246/642 180/642 0/6 | 438/648 246/642 180/642 0/6 0/6 | 438/648 246/642 180/642 0/6 0/6 |
| 4 | 876/1296 492/1284 360/1284 0/12 | 876/1296 492/1284 360/1284 0/12 0/12 | 876/1296 492/1284 360/1284 0/12 0/12 |
| 16 | 5088/7008 | 5088/7008 | 5088/7008 |

One stream's depths follow each round's timing, so they move between sessions in both arms; its kept rows (535, 539,
538 code; 441, 440, 457 chat) do not regress. At sixteen admitted streams the depth rule drafts one row.

## Against the other engines

`engines.sh` (`--levels 1,4,64 --reps 3 --tokens 256 --temperatures 0`; TensorFold `--context 16384 --parallel 64`,
the window that fits 64 lanes), end-to-end tok/s, medians of three (a reply that arrives as one text piece has no
decode time, so only end to end compares):

| code / chat | TensorFold, DSpark | oMLX 0.8.2, MTP | ratio |
|---|---|---|---|
| 1 stream | **113.7 / 86.2** | 13.5 / 36.2 | 8.4x / 2.4x |
| 4 streams | **293.9 / 240.0** | 39.2 / 48.3 | 7.5x / 5.0x |
| 64 streams | **344.3 / 236.2** | 200.0 / 184.5 | 1.7x / 1.3x |

- oMLX serves its own 4-bit build (`Inkling-2-Mini-oQ4e-mtp`, about 17 GB) with MTP drafting on; TensorFold serves the
  official NVFP4/MXFP8 checkpoint. oMLX ran at load average 70-90; in this branch's first session (load 14-40) it
  reached 45.1 / 8.3, 107.0 / 79.1 and 209.3 / 227.6, still under ours at every cell but 64 chat streams (1.02x).
- #4618 on this Mac (its own clients, `--parallel 4 --context 16384`): oMLX 0.8.1 with DSpark 72.5 tok/s on one chat
  stream and 126.6 on four; TensorFold's #4618 head 123.4 code / 91.5 chat on one stream, 281.1 / 223.0 on four.
- Strata and llama.cpp were not measured here: neither is installed, and running them needs their installers and
  weights (Strata's app and SQ packs; an Inkling GGUF and a llama.cpp build with the looped-layer support) put on this
  Mac, which is the owner's call. Strata's own model card publishes, for an M5 Max 128 GB, 100.0-114.5 tok/s on one
  request and 597.2 with 64. The one-request figure is under ours; the 64-request figure is above ours (344.3 code /
  236.2 chat) on a smaller chip, so at 64 streams the fastest-engine line stays open until Strata runs on this Mac.

## Exactness

`TF_INKLING2_METAL_DIR` and `TF_INKLING2_DSPARK_DIR` set to the checkpoint, the `test-inkling2-metal-host` test
binary runs on the device:

- `Mini Metal DSpark sixteen-stream chain preserves solo holds at unequal depths`: depths 1-7, four streams then all
  sixteen in one hold, each equal to its own solo hold;
- `Mini Metal DSpark four prompts equal solos through a leave and join`: four greedy replies equal four solo replies
  token for token while a stream leaves and another joins;
- `Mini Metal opt-in DSpark drafting preserves target tokens and rolls back exactly`.

## Kernel contract

`tf-kernel-contract --only 'dspark cohort'` on this Mac: `SUMMARY 5 kernels: 5 pass gates 1-4, 0 fail, 0 known
findings; 5 controls: 5 red as planted, 0 missed; 0 limit problems`. Each cohort launch
(`zig/tests/contract/metal_inkling2_dspark_cohort.zig`, records in `inkling2_dspark_cohort_metal_records.zig`) is
checked byte for byte against the solo launches it replaces run a stream at a time on the same inputs: `block_qkv`
at each anchor, `recurrent` at each kappa row and held token, `chain_pick`, `logits` per base row, and the host's
row copies for `pack`. Each control plants one stream's or row's wrong index and turns red at `shapes`.

## Files

`cells.jsonl`: each cell of the five sessions; `bench.sh` and `engines.sh` reproduce them (`MODEL` names the
checkpoint, `OUT` the scratch folder, `OMLX` the oMLX binary).
