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
run ends with one `RESULT` line for each engine, level and cell, and one `FASTEST` line for each level and cell. A
`FASTEST` line counts only cells with no failed or unmeasured request, and only among the engines that ran.

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

Aggregate tok/s, median of 3, 256 tokens a request, temperature 0:

| Streams | Cell | TensorFold | llama.cpp IQ4_NL | Strata SQ-20 |
|---|---|---|---|---|
| 1 | same | **94.3** | 35.9 | unmeasured (3 of 3) |
| 1 | diverse | **129.5** | 32.1 | (659.5, see below) |
| 4 | same | **232.5** | 95.4 | unmeasured (12 of 12) |
| 4 | diverse | **177.1** | 101.3 | (138.1, see below) |
| 64 | same | **485.5** | 394.8 | unmeasured (102 requests) |
| 64 | diverse | **405.7** | 312.0 | unmeasured (19 requests) |

Every TensorFold and llama.cpp cell finished with no failed or unmeasured request.

**Fastest engine measured on this Mac: TensorFold, at 1, 4 and 64 streams, on both cells.** That compares TensorFold
with llama.cpp. Strata was not timed: it is neither slower nor faster here.

Strata's figures are not timings. Its server finished every request with 256 tokens (`done: 256 tokens`, 420 in
`server.log`). But it streamed the thinking text in a lump, so `bench_concurrent.py` saw no timed tokens in the "same" cells
and on most 64-stream requests. The 659.5 tok/s is one burst after a 2.5 s first token, not a decode rate. Strata's
own log puts single-stream SQ-20 at 67 to 99 tok/s; that figure is the engine's own count and was not checked by our client.

Settings that differ between engines (see the table above):

- Weights: TensorFold runs the official `inkling2` checkpoint, and llama.cpp runs an IQ4_NL re-quantization.
- KV: TensorFold uses bf16 and llama.cpp uses f16.
- Drafting: TensorFold drafts with DSpark, which accepted 9/23 to 154/498 of its drafts in the server log. llama.cpp
  runs no drafter.
- Thinking: TensorFold thinks on every request (`thinking=True`). Strata logged `thinking=False` for its requests
  but still produced thinking text.

This is one Mac, one session, one model. These figures do not compare chips.
