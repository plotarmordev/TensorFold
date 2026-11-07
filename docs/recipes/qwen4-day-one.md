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

1. Update and build: `git pull`, then `zig build test test-golden` and `zig build test-qwen4-metal-sources test-qwen4-metal`.
2. Pull the BF16 release with its revision: `hf download Qwen/<repo> --revision <sha> --local-dir ~/models/qwen4-hf`.
   Read `model_type`, `num_experts`, `mtp_num_hidden_layers` (or `mtp`) and `quantization_config` from `config.json`.
3. Fit before converting. Affine 6-bit in groups of 32 with BF16 scales and biases stores 7 bits a weight,
   0.875 GB per billion parameters; add the n-gram tables (29.8 GiB on Flash Next), KV/state and the prompt cache.
   Compare with `sysctl iogpu.wired_limit_mb`; raise it with `sudo sysctl iogpu.wired_limit_mb=<MiB>` when the plan names it.
4. Convert the Flash tier to MLX affine 6-bit/group 32, the only width the Flash Next Metal kernels read, keeping
   the `language_model.` names: `python -m mlx_vlm.convert --hf-path ~/models/qwen4-hf --mlx-path ~/models/qwen4-6bit -q --q-bits 6 --q-group-size 32`.
   Check the output index still lists `mtp.*` or `language_model.mtp.*` tensors; a conversion without them serves without drafts.
5. Check the plan: `tensorfold info ~/models/qwen4-6bit` prints the load and cold pack budgets,
   `tensorfold plan ~/models/qwen4-6bit` the verdict against this Mac's memory. Do not serve a refused plan.
6. Serve: `tensorfold serve ~/models/qwen4-6bit --name bench --context 32768 --temperature 0 --no-thinking`.
   The first open builds packs under `~/.cache/tensorfold/packs/`; a refusal names the config field or tensor to report.
7. Measure: `python3 tools/bench_openai.py http://127.0.0.1:8080 bench --tokens 64 --reps 5 --temperatures 1.0,0 --output bench.json`,
   once with drafts and once with `--no-drafts`, comparing greedy output hashes; then `tensorfold benchmark ~/models/qwen4-6bit`.
   Record the checkpoint revision, converter version, engine commit, peak memory and launch command with the result.
