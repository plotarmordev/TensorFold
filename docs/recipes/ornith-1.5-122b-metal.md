# Ornith-1.5-122B on Metal (one 256 GB M5 Ultra)

The native Metal engine serves [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a Qwen3.5-MoE
(`qwen3_5_moe`: 48 layers, Gated DeltaNet with gated attention every fourth layer, 256 routed experts of 1,024 with
eight picked, a shared expert) through the 27B's engine (`zig/src/families/qwen27/`, #1125), from the publisher's
MLX 4-bit weights, with incoai's DFlash2 drafter. Receipts are in [docs/receipts/ornith-122b-metal](../receipts/ornith-122b-metal/README.md).

## Checkpoint

| Repository | Revision | On disk |
| --- | --- | ---: |
| [`ornith-ai/Ornith-1.5-122B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit) | `06a57aba4fbae218efc9716bcdf7d6b3035331c8` | 64.85 GiB (63.99 GiB in the 14 text shards) |
| [`incoai/Ornith-1.5-122B-DFlash2`](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) | `9baefc1bec85c08193ab442e0ca5f7057669b61d` | 1.7 GiB |

The MLX weights are affine 4-bit in groups of 64, with the router and the shared-expert gate at 8 bits.

## Run

```sh
tensorfold-native serve ~/models/Ornith-1.5-122B-MLX-4bit --drafter ~/models/Ornith-1.5-122B-DFlash2 \
  --name bench --parallel 4 --context 32768
```

`--no-drafts` decodes one token a round. Loading takes about 10 s from a warm page cache (55 s cold). Between requests,
`/health`'s `memory.active` reports 70.6 / 71.1 GiB without drafts and 71.6 / 72.1 GiB with DFlash2 at one / four
streams of 32,768 tokens, out of the 179 GiB the 70% cap allows on 256 GiB, plus a 16 GiB prompt cache by default.
Idle and peak health snapshots are in [health.txt](../receipts/ornith-122b-metal/health.txt).

## Speed

Greedy, 256 tokens per request, the first four of `tools/bench_prompts.json`'s 16 prompts (request `i` takes prompt
`i`: `quicksort-raw`, `rust-reverse-words`, `multiply-17-23`, `sleep-tips`), median of three. Each server was a fresh
boot. "Peak" is `/health`'s `memory.peak` after the run.

| Drafts | Streams | Server | tok/s per stream | Aggregate tok/s | First token (median, max) | Tokens a round | Peak |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| none | 1 | `--parallel 1` | 91.5 | 91.5 | 0.041 s, 0.042 s | 1.0 | 71.1 GiB |
| none | 1 | `--parallel 4` | 89.4 | 89.4 | 0.038 s, 0.05 s | 1.0 | |
| none | 4 | `--parallel 4` | 53.6 | 211.8 | 0.131 s, 0.26 s | 1.0 | 74.6 GiB |
| DFlash2 | 1 | `--parallel 1` | 241.8 | 241.8 | 0.043 s, 0.043 s | 6.2 | 72.2 GiB |
| DFlash2 | 1 | `--parallel 4` | 239.3 | 239.3 | 0.040 s, 0.05 s | 6.2 | |
| DFlash2 | 4 | `--parallel 4` | 76.1 | 254.9 | 0.110 s, 0.25 s | 3.3 | 76.7 GiB |

DFlash2 drafts for one stream at a time (the server says `DFlash2 drafts for one of them`), so its four-stream gain
over plain decoding is 1.20x against 2.64x on the dedicated one-stream servers (2.68x at one stream on the
four-stream servers). Every drafted reply's token SHA-256 equals its `"draft": false` run, and every concurrent
reply equals its solo run.

## Quality against the publisher's reference

The same prompt ids through mlx-lm 0.32.2 (MLX 0.32.4) on the same weights, greedy, 128 tokens each:

- Two raw-code completions (`quicksort-raw`, `js-debounce-raw`) match all 128 tokens; the third (`c-primes-raw`)
  splits at token 18. Over the 16 prompts the leading match is 636 of 2,048 tokens.
- The first split in each other prompt is routing-sensitive: under mlx-lm's own full forward over the
  shared prefix, its token leads TensorFold's by -4.1 to 4.1 nats, and mlx-lm prefers TensorFold's token at 5 of the
  14 splits. A 4-bit mixture of experts' routing flips between kernels with different bf16 accumulation, after which
  greedy replies diverge.
- Scored by mlx-lm's teacher-forced forward of the publisher's weights, TensorFold's continuations have a mean NLL of
  0.1126 a token, mlx-lm's own 0.1187: TensorFold's replies are as likely under the reference as the reference's own.

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

Plain serving does not load the layer and is unchanged: on fresh `--parallel 1 --no-drafts` servers, alternating
`6d4811726` and this change twice, the 16 reference prompts' greedy 128-token replies are identical (2,048 token ids),
and one stream decodes at 90.8 / 85.2 tok/s before and 88.4 / 85.3 tok/s after (`tools/bench_concurrent.py --levels 1`,
256 tokens, three repetitions).

## Not covered

- MTP drafting in `serve`. The layer drafts once a step in the engine, but `serve` does not draft or verify with it
  yet; plain serving does not load it.
- Images, sampled replies and prompts longer than the bench's (under 50 tokens) were not measured here.
