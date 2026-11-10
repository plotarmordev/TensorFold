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

Plain serving does not load the layer and is unchanged: on fresh `--parallel 1 --no-drafts` servers, alternating the
base and this change twice, the 16 reference prompts' greedy 128-token replies are identical (2,048 token ids), and one
stream decodes at 90.8 / 85.2 tok/s before and 88.4 / 85.3 tok/s after (`tools/bench_concurrent.py --levels 1`, 256
tokens, three repetitions).

## Not covered

- MTP drafting. The layer loads and runs, but `serve` does not draft with it yet; plain serving does not load it.
