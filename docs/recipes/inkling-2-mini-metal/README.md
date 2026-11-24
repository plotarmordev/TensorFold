# Inkling 2 Mini on the M5 Ultra: TensorFold, llama.cpp and Strata at 1, 4 and 64 streams

`engines.sh` measures each engine on this Mac's Metal GPU, one at a time, at 1, 4 and 64 concurrent streams. Every
engine uses the same client, prompts and token counts as our own bench (`docs/receipts/inkling-2-mini-metal-210/bench.sh`).
It needs a Mac GPU window (`/Users/Shared/agent-work/mac-gpu-window`) and inputs the Desk has staged. It downloads and
installs nothing.

    sh docs/recipes/inkling-2-mini-metal/engines.sh [tensorfold|llama.cpp|strata-sq20|strata-sq10 ...]

## What the Desk stages

Everything goes in `STAGE`, which defaults to `/Users/agents/projects/capri/desk/jobs/tf250-engines`.

| Path in `STAGE` | Source | Size | sha256 |
|---|---|---|---|
| `llama.cpp/` | `llama-bNNNNN-bin-macos-arm64.tar.gz` from ggml-org/llama.cpp, unpacked; needs build b12406 or later (Inkling 2 arrived in #32217); staged: b12655 (commit `969d73331`, tarball `03278e8e…2ae0`) | | its release digest |
| `gguf/Inkling-2-Mini-IQ4_NL.gguf` | [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) | 17,627,940,768 B | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |
| `strata/strata-inkling` | `strata-inkling-0.1.65-macos-arm64-metal.tar.gz`, Strata v0.2.19 (tarball `fa29cf5d…f3f`) | | `9bd678185e41526eb3796220c3dc7723bb2d13b60e585241ee4f05e7c492f27b` |
| `strata/control/` | `strata-control-0.1.65-macos-arm64.tar.gz`, Strata v0.2.19 | | tarball `c5ad7e815929ea2076cfc66510ac142727d15de3b17f5256bc15aff1a74eddbc` |
| `strata/tokenizer.json` | the pack repo's tokenizer, used when a pack embeds none | | `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6` |
| `strata/Inkling-2-Mini-SQ-20.strata`, `-SQ-10.strata` | [strata-engine/Inkling-2-Mini-Strata](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata) @ `04996a1e` | 19,346,685,952 / 10,471,526,400 B | `0bbfd523…7e1b97` / `ac4d452a…065934` |

Strata v0.2.19 ships a Metal engine. `strata-inkling 0.1.65` reports `"backend": "metal"` and `macos-arm64` in its
`BUILD.json`. The Linux GB10 copy staged for card 242 does not run on this Mac.

The script checks every pinned digest. It prints the GGUF's path, size and sha256, and refuses any llama.cpp build
older than b12406. Homebrew's `llama-server` (build 12359) cannot load Inkling 2.

## Settings

The settings are the same for every engine:

- Client: `tools/bench_concurrent.py`.
- Output: 256 tokens a request, temperature 0.
- Runs: 3 reps, and the median is reported.
- Server: one server with 64 slots for all three levels.
- Timeout: each level and cell is one run under `timeout 600`. macOS has no `timeout`, so a perl `alarm` stands in.
- Prompts: chat only, because Strata's server has no `/v1/completions`. Each level has two cells:
  - **same**: bench_concurrent's built-in chat prompt on every stream.
  - **diverse**: request *i* gets chat prompt *i* of the 13 in `tools/bench_prompts.json`.

  On every stream, the same prompt sends all 64 greedy streams through the same experts, so the diverse cell is the fairer
  64-stream figure.

The engines differ in these settings, and the receipt prints each one:

| | Weights | KV | Drafting | Flags |
|---|---|---|---|---|
| TensorFold (feat/210 head) | official checkpoint, `inkling2` quant (`~/models/Inkling-2-Mini`) | bf16, Metal's only | DSpark (`--drafter`) | `serve --context 65536 --parallel 64` |
| llama.cpp | IQ4_NL GGUF, a re-quantization (95.5% top-token agreement with BF16 on its card) | f16 | none | `-ngl 99 -fa on -ctk f16 -ctv f16 -c 262144 -np 64 --jinja` |
| Strata SQ-20 / SQ-10 | Strata's own packs | `--kv auto`, read from the server log | DSpark (its default) | Control's `server.py`, `parallel 64`, `--max-context 32768` |

Each engine logs `setup: N s` and `measure: N s` separately, to `OUT/ENGINE.log` (default `/tmp/ink2-engines`). The
run ends with one `RESULT` line for each engine, level and cell, and one `FASTEST` line for each level and cell.
`FASTEST` ranks by bench_concurrent's `end_to_end_tps`: the usage's output tokens over the cell's whole makespan. This
rate needs no stream timing, so it times Strata, whose stream drops the thinking (see the receipt). A `FASTEST` line
counts only cells where every request completed with usage, and only the engines this invocation completed.

## Receipt

Measured on 25 Nov 2026, 01:46 to 02:45 HKT, in the Desk's Mac GPU window ("250 only"). The Desk staged every input
above on 25 Nov 01:1x (`tf250-engines/setup.txt`: "TF 250 SETUP DONE", digests checked). The run used the GGUF at
`STAGE/gguf/Inkling-2-Mini-IQ4_NL.gguf` (17,627,940,768 B, sha256 `cbdf3fa9…81af4`, checked by the script) and llama.cpp
b12655.

There were two runs:

- **Run 1, `engines.sh` at `0962da4a`.** Only TensorFold's cells count. Its server outlived its turn because the exit
  trap read a `local` pid after `run` returned. llama.cpp and both Strata servers then could not bind port 18250
  (their logs say so), so their cells measured TensorFold again. `737b4a6b` fixes the trap and refuses a busy port.
  Job logs: `logs/run1/` (the engine's phase log and one bench log for each cell).
- **Run 2, `engines.sh` at `737b4a6b`, `llama.cpp strata-sq20 strata-sq10`.** llama.cpp took 66 s to set up and
  418 s to measure; Strata SQ-20 took 109 s and 296 s. The window closed during Strata SQ-10's setup, so SQ-10 is not
  measured. Job logs: `logs/run2/`, including Strata's own `done:` lines (`strata-sq20/server-done.log`).

TensorFold's engine is the feat/210 head (`d29431d5f`); the commits after it change only this folder. Its quant is
`inkling2`: NVFP4 1×16 and MXFP8 32×32 groups, the rest bf16. Run 1 printed the quant as blank because the script
read a missing key; `737b4a6b` reads `quant_method`.

Tok/s end to end, median of 3. Each request asks for 256 tokens at temperature 0. In brackets: the streamed rate, our
bench's usual figure.

| Streams | Cell | TensorFold | llama.cpp IQ4_NL | Strata SQ-20 |
|---|---|---|---|---|
| 1 | same | **92.6** (94.3) | 35.4 (35.9) | 85.7 (unmeasured) |
| 1 | diverse | **125.1** (129.5) | 31.8 (32.1) | 89.1 (burst) |
| 4 | same | **229.2** (232.5) | 93.4 (95.4) | 134.7 (unmeasured) |
| 4 | diverse | **175.9** (177.1) | 100.1 (101.3) | 96.5 (burst) |
| 64 | same | **486.2** (485.5) | 384.1 (394.8) | 382.5 (unmeasured) |
| 64 | diverse | **406.6** (405.7) | 291.3 (312.0) | 346.1 (unmeasured) |

**Fastest engine measured on this Mac: TensorFold, at 1, 4 and 64 streams, on both cells**, ahead of llama.cpp b12655
IQ4_NL and Strata v0.2.19 SQ-20. No request failed. Strata SQ-10 was not measured.

Every TensorFold and llama.cpp reply ran the full 256 tokens. Strata ignores `ignore_eos`, so 81 of its 207 diverse
replies ended before 256 tokens (`finished` in the bench logs). Its end-to-end rate counts the tokens it
did produce, so it is still a rate, but over shorter replies.

Strata's streamed figures are not timings. Strata control 0.1.65 forces `reasoning_effort` to `none`
(`serve/inkling_dialect.py:282`), yet the model still thinks. The control then silently discards every
`<|content_thinking|>` span (`:189`), so the stream carries only the text after the thinking. `bench_concurrent.py`
sees too few pieces to time. All "same" cells and some 64-stream diverse requests are unmeasured, and the 1- and
4-stream diverse figures are bursts after a 2.5 s first token. No request setting changes that, so the verdict uses the
end-to-end rate. `engines.sh` printed the streamed rates in the run; the end-to-end column was read from the same
result files at `logs/run2/*/*.log` after the run, with the summary `engines.sh` now prints.

Settings that differ between engines (see the table above):

- Weights: TensorFold runs the official `inkling2` checkpoint, and llama.cpp runs an IQ4_NL re-quantization.
- KV: TensorFold uses bf16 and llama.cpp uses f16.
- Drafting: TensorFold drafts with DSpark, which accepted 9/23 to 154/498 of its drafts in the server log. llama.cpp
  runs no drafter.
- Thinking: TensorFold thinks on every request (`thinking=True`). Strata logged `thinking=False` for its requests
  but still produced thinking text.

This is one Mac, one session, one model. These figures do not compare chips.
