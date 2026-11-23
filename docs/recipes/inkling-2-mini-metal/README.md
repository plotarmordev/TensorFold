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
| `llama.cpp/` | `llama-bNNNNN-bin-macos-arm64.tar.gz` from ggml-org/llama.cpp, unpacked; needs build b12406 or later (Inkling 2 arrived in #32217); b12651 is current | | its release digest |
| `Inkling-2-Mini-IQ4_NL.gguf` | [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) | 17,627,940,768 B | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |
| `strata/strata-inkling` | `strata-inkling-0.1.65-macos-arm64-metal.tar.gz`, Strata v0.2.19 (tarball `fa29cf5d…f3f`) | | `9bd678185e41526eb3796220c3dc7723bb2d13b60e585241ee4f05e7c492f27b` |
| `strata/control/` | `strata-control-0.1.65-macos-arm64.tar.gz`, Strata v0.2.19 | | tarball `c5ad7e815929ea2076cfc66510ac142727d15de3b17f5256bc15aff1a74eddbc` |
| `strata/tokenizer.json` | the pack repo's tokenizer, used when a pack embeds none | | `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6` |
| `Inkling-2-Mini-SQ-20.strata`, `-SQ-10.strata` | [strata-engine/Inkling-2-Mini-Strata](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata) @ `04996a1e` | 19,346,685,952 / 10,471,526,400 B | `0bbfd523…7e1b97` / `ac4d452a…065934` |

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

None yet. Card 250 has no Mac GPU window and its setup has not been staged:

- The Desk's `tf250-engines/setup.txt` says llama.cpp ≥ b12406 is not built, and there is no Inkling 2 GGUF on the Mac.
- The disk is under its free-space floor.

Nothing here claims an engine is faster until `engines.sh` has run in a window.
