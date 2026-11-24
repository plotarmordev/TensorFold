# Strata v0.2.19, Inkling 2 Mini on GB10: card 247 receipt

**Not counted.** SQ-10 at 64 streams under Strata's own `--bench` ran 442.8 tok/s, 55% of the 805.8 its pack card
publishes (the gate wants 90-110%). Nothing below is a bar for TensorFold; the numbers say where Strata stands with
the inputs and flags we could pin.

## Runs

| What | Job | Commit | Machine |
| --- | --- | --- | --- |
| Own `--bench`, both packs, 1/4/64 streams | `bottom-c5e4253968cb` | card 242's `strata-job.sh` | BOTTOM GB10 |
| HTTP bench, SQ-20, 1/4/64 streams | `bottom-8e4de9ca491c` (measure 351 s) | `1e17bb6ca` | BOTTOM GB10 |
| HTTP bench, SQ-10, 1/4/64 streams | `bottom-93a2de938e6e` (measure 269 s) | `1e17bb6ca` | BOTTOM GB10 |

The Desk ran every job on BOTTOM in DeepSeek windows, not on TOP, at the Lead's call (card 242, 24 Nov). Full logs:
Desk `jobs/tf247-strata/sq20-1e17.log`, `sq10-1e17.log`. Scripts: `strata-sq{20,10}-{1,4,64}.sh` and
`strata-http-sq{20,10}.sh`, all through `strata-job.sh`, which prints every setting below as `[settings]` lines.

## Results (tok/s total across streams, median of 3)

| Pack | Streams | Own `--bench` decode | HTTP end to end, same prompt | HTTP end to end, different prompts | Published |
| --- | ---: | ---: | ---: | ---: | ---: |
| SQ-20 | 1 | 23.8 | 38.1 | 38.0 | 46.1 |
| SQ-20 | 4 | 75.4 | 62.9 | 57.6 | |
| SQ-20 | 64 | 332.1 | 204.6 | 180.8 | 1,145.1 |
| SQ-10 | 1 | 57.5 | 53.4 | 53.4 | 74.3 |
| SQ-10 | 4 | 94.2 | 79.3 | 74.2 | |
| SQ-10 | 64 | 442.8 | 269.3 | 240.7 | 805.8 |

Own bench: decode only, the `[bench]` line's `decode=… tps agg`. HTTP: every stream's usage tokens over the makespan
(first request sent to last reply done), prefill included, so it runs below a decode rate. No HTTP cell failed a
request.

## Settings

- Engine: Strata v0.2.19 = `strata-inkling 0.1.65`
  ([release](https://github.com/Niko1221/Strata/releases/tag/v0.2.19)), packs checked against the digests in its
  `inkling_family.py`. Its published bench flags, from `ink_main.cpp`'s `--help` at that tag: "--bench --native PATH
  [--parallel 1..64] [--gpus 1|2] [--ep-split F] [--vram-reserve-mib N] [--cpu-threads N] [--kv
  auto|fp16|fp8|k8v8|k8v4] [--prompt-len N] [--gen N]".
- Published figures: the [pack card](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata), "How speeds were
  measured": "| DGX Spark GB10 128 GB | SQ-20 / SQ-10 | 2,197 / 2,070 | 46.1 / 74.3 | 1,145.1 / 805.8 with 64 |",
  "Measured with thinking off". It names no engine version, flags, prompt or context.
- Own bench: `--bench --native PACK --parallel N`, nothing else set.
  - `--parallel N`: N slots, each decoding one token a step in one batched forward.
  - KV dtype: `--kv auto`, fp16 below 16 slots (the 4-slot lines print `kv=fp16`), fp8 from 16.
  - Prompt: 64 synthetic token ids a slot, id `100 + (31*slot + position) % 1000`: one ramp shifted 31 ids a slot,
    so each stream's prompt differs; Strata's bench has no same-prompt mode. Output: 32 greedy tokens a slot.
  - Graphs: the engine's default, no flag. Drafting: DSpark on (default), its tree at 1 slot only.
  - Warmup: none; each run is a fresh process whose one pass is timed.
- HTTP bench: Strata's server (`serve/server.py`, strata-control 0.1.65), config `"parallel": N`,
  `"drafting": "dspark"`, `--max-context 32768`, under `tools/bench_concurrent.py`.
  - N requests at once on `/v1/chat/completions`, 128 output tokens asked, temperature 0, `ignore_eos`, `--reps 3`,
    one 16-token warm-up request a cell.
  - Same prompt: the first chat prompt of `tools/bench_prompts.json` (31 tokens) on every stream. Different prompts:
    request i gets chat prompt i mod their count (21-47 tokens).
  - KV dtype: not recorded by these runs. The server logs no `kv_dtype=` line (the engine reports it only in its
    `INFO` protocol line, which the server keeps in `/metrics`), so the jobs at `1e17bb6ca` printed none. Control
    passes no `--kv`, so the engine's auto rule applies: fp16 at 1 and 4 slots, fp8 at 64 `[INFERENCE]`, from
    `ink_main.cpp` at v0.2.19. `strata-job.sh` now prints a `[kv]` line per pack and level from `/metrics` and fails
    when the dtype is absent; a rerun records it.

## What still differs from the published 805.8

- Flags: the card gives none. Our own bench runs the engine's defaults (64-token prompt, 32 output tokens, `--kv
  auto`); the card's prompt and output lengths are unknown, so its decode window may differ from ours.
- Path: the card's "many at once" is "the total written across that many requests running together", i.e. served
  requests with drafting, while `--bench` at 64 slots takes the plain batched path (DSpark's tree runs at 1 slot
  only). Our HTTP cells are the served path, but under the thinking fault below, so neither side matches the card's
  workload.
- Thinking: the card measures "with thinking off". Strata's server renders thinking off, yet the model still opens
  `<|content_thinking|>` and the server drops that span, so the HTTP replies are mostly empty text (`no_output`:
  SQ-10 at 64 same prompt 192 of 192, SQ-20 141 of 192 plus 51 truncated). Tokens are generated and counted, so the
  end-to-end rate stands, but there is no per-stream decode rate and the replies are not usable answers.
- Machine: BOTTOM, not TOP; both are GB10. The jobs ran in the Desk's DeepSeek windows on BOTTOM.

## FEED

- FEED 211: Strata v0.2.19 Inkling 2 Mini on GB10 (bottom-c5e4253968cb, own --bench, decode, median of 3): SQ-10
  57.5 / 94.2 / 442.8 tok/s at 1 / 4 / 64 streams, SQ-20 23.8 / 75.4 / 332.1; 55% of the published 805.8, not a bar.
- FEED 238: Strata v0.2.19 HTTP end to end on GB10 (bottom-93a2de938e6e, bottom-8e4de9ca491c, median of 3, same /
  different prompts): SQ-10 53.4/53.4, 79.3/74.2, 269.3/240.7; SQ-20 38.1/38.0, 62.9/57.6, 204.6/180.8 tok/s at
  1 / 4 / 64 streams; replies mostly thinking-only, not a bar.
- FEED 240: Strata's published GB10 figures do not reproduce with v0.2.19's default `--bench` (bottom-c5e4253968cb,
  decode, median of 3): SQ-10 57.5 at 1 stream and 442.8 at 64 against 74.3 / 805.8; SQ-20 23.8 at 1 and 332.1 at
  64 against 46.1 / 1,145.1. Its card names no flags or version, so they are not a target.
- FEED 241: Strata v0.2.19's server emits no answer text for Inkling 2 Mini with thinking off (the model opens
  `<|content_thinking|>`, the server drops it): at 64 streams, same prompt, SQ-10 192 of 192 replies `no_output`
  (bottom-93a2de938e6e), SQ-20 141 `no_output` and 51 truncated of 192 (bottom-8e4de9ca491c); at 1 and 4 streams,
  same prompt, every reply `no_output`.
