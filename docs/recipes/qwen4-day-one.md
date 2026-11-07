# Qwen 4 on day one

What to run on this Mac (M5 Ultra, 256 GB) the day the first Qwen 4 checkpoint is published.
No Qwen 4 checkpoint existed when this was written; the routes below are what the engine admits from synthetic configs.

## Which engine reads it

| `model_type` in `config.json` | Engine | Notes |
| --- | --- | --- |
| `qwen4_moe` (Flash MoE tier, expected after Qwen3.5's `qwen3_5_moe`) | Flash Next Metal (`qwen4_exp`) | Routed experts and an MTP head, nested `mtp` block or flat `mtp_num_hidden_layers`. |
| `qwen4_exp` | Flash Next Metal | As [Qwen3.8 Flash Next](qwen3.8-flash-next.md). |
| `qwen4`, `qwen4_text` | Qwen4 dense | [Dense preparation](qwen4-dense.md); refuses MoE fields and a positive MTP head by name. |

Any other tag stops at startup with "no backend for ...": report it with the config.

## Checklist

1. Update, test and build the native binary; every command below runs that binary:

   ```sh
   git pull && zig build test test-golden test-qwen4-metal-sources test-qwen4-metal
   zig build native -Dcpu=apple_m1
   tf=$PWD/zig-out/native/bin/tensorfold-native
   ```

2. Pull the BF16 release with its revision: `$tf pull Qwen/<repo>@<sha>`, or `hf download Qwen/<repo> --revision <sha> --local-dir ~/models/qwen4-hf`.
   Read `model_type`, `num_experts`, `mtp_num_hidden_layers` (or `mtp`) and `quantization_config` from `config.json`.
3. Convert the Flash tier to MLX affine, keeping the `language_model.` names. 6-bit in groups of 32 is the recommended
   width (Flash Next's qualified one); Metal also reads 2, 3, 4, 5 and 8 bits in groups of 32, 64 or 128:
   `python -m mlx_vlm.convert --hf-path ~/models/qwen4-hf --mlx-path ~/models/qwen4-6bit -q --q-bits 6 --q-group-size 32`.
   Check the output index lists the head's `mtp.*` or `language_model.mtp.*` tensors when `config.json` declares one:
   the Flash Next Metal loader requires every declared head tensor, and a conversion that drops them is refused.
4. Fit: `$tf info ~/models/qwen4-6bit` prints the `startup budget` and the `cold pack budget` (the first open, before a pack is cached).
   Both must sit under the load ceiling: 70% of RAM (179.2 GiB on this Mac), capped by the GPU's recommended working set.
   `FLASHNEXT_LOAD_LIMIT_GB` raises it to at most that working set; a larger limit can stall macOS under wired memory.
   Over the ceiling, convert to fewer bits, or serve with `--ple-on-ssd` and `--ssd-experts GIB` ([residency](fluid-subgraph-residency.md)).
5. Serve: `$tf serve ~/models/qwen4-6bit --name bench --context 32768 --temperature 0 --no-thinking`.
   The first open builds packs under `~/.cache/tensorfold/packs/`; a refusal names the config field or tensor to report.
6. Measure: `python3 tools/bench_openai.py http://127.0.0.1:8080 bench --tokens 64 --reps 5 --temperatures 1.0,0 --output bench.json`,
   once with drafts and once with `--no-drafts`, comparing greedy output hashes; then stop the server and run `$tf benchmark ~/models/qwen4-6bit`.
   Record the checkpoint revision, converter version, engine commit, peak memory and launch command with the result.
