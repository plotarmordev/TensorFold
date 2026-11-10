# Ornith-1.5-122B on Metal: the MTP layer

The native Metal engine serves [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a Qwen3.5-MoE
(`qwen3_5_moe`: 48 layers, 256 routed experts with eight picked, a shared expert), through the 27B's engine
(`zig/src/families/qwen27/`) from the publisher's MLX 4-bit weights
([`ornith-ai/Ornith-1.5-122B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit) `06a57ab`).
The MLX release leaves out the checkpoint's multi-token-prediction (MTP) layer; this page covers loading and running it.

## MTP layer

`Model.loadMtp` reads the `mtp.*` tensors from the bf16 release's last two shards
([`ornith-ai/Ornith-1.5-122B`](https://huggingface.co/ornith-ai/Ornith-1.5-122B) `7681ef6`, shards 61 and 62 of 62;
the folder needs only those, `config.json` and `model.safetensors.index.json`) and quantizes every matrix (fc,
attention, the shared expert, router and all 256 experts) to MLX affine 4-bit in groups of 64 on the host at load, the
norms shifted to `1 + w`. Its forward runs as one more full-attention block past the target's 48, with its own K/V cache.
`tools/zig/ornith_mtp_reference.py` writes the test's fixture through mlx-lm's `DecoderLayer` on the same 4-bit weights;
every one of its 11 rows matches TensorFold's at a cosine of 0.996 or better:

```sh
TF_QWEN27_DIR=~/models/Ornith-1.5-122B-MLX-4bit TF_QWEN27_MTP_DIR=~/models/Ornith-1.5-122B-mtp zig build test-qwen27-mtp
```

`qwen27.mtp.Drafter` runs the layer once a decode step: the runner taps the target's last layer before its final
norm, and after each target pass the layer reads those rows beside each next token at the same cache rows, writing its
keys and values to its own cache, then drafts through the target's head into its own logits row.
`tools/zig/ornith_mtp_drafts.py` writes the second test's fixture: mlx-lm's greedy 128-token reply to each of
`tools/bench_prompts.json`'s 16 prompts, and its MTP layer's draft at each step. Teacher-forced on those replies,
TensorFold's 2,048 drafts equal mlx-lm's at 1,900; mlx-lm's own drafts one step at a time with a cache equal its
whole-pass ones at 1,917. Every miss is within 3.4 logits of the reference's runner-up, and at both misses above
1.5 mlx-lm's stepped drafts split from its whole pass as well.

Plain serving does not load the layer and is unchanged: on fresh `--parallel 1 --no-drafts` servers, alternating the
base and this change twice, the 16 reference prompts' greedy 128-token replies are identical (2,048 token ids), and one
stream decodes at 90.8 / 85.2 tok/s before and 88.4 / 85.3 tok/s after (`tools/bench_concurrent.py --levels 1`, 256
tokens, three repetitions).

## Not covered

- MTP drafting in `serve`. The layer drafts once a step in the engine, but `serve` does not draft or verify with it
  yet; plain serving does not load it.
