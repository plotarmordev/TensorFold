# Ornith-1.5-122B on Metal (one 256 GB M5 Ultra)

**Serve with DFlash2: 244 tok/s on one stream (2.6× plain) and 252 tok/s across four (1.3× plain), same tokens as plain.**
MTP keeps plain's tokens but trails DFlash2: at four streams it gives 213.7 tok/s against DFlash2's 252.3 (plain
191.0), and a lone MTP stream stops drafting after its first requests (see **Speed**), so it is not the default.

The native Metal engine serves [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a Qwen3.5-MoE
(`qwen3_5_moe`: 48 layers, Gated DeltaNet with gated attention every fourth layer, 256 routed experts of 1,024 with
eight picked, a shared expert) through the 27B's engine (`zig/src/families/qwen27/`, #1125), from the publisher's
MLX 4-bit weights, with incoai's DFlash2 drafter. Receipts are in [docs/receipts/ornith-122b-metal](../receipts/ornith-122b-metal/README.md)
and, for the 13 November speed and quality figures, [docs/receipts/ornith-122b-metal-mtp](../receipts/ornith-122b-metal-mtp/README.md).

## Checkpoint

| Repository | Revision | On disk |
| --- | --- | ---: |
| [`ornith-ai/Ornith-1.5-122B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-122B-MLX-4bit) | `06a57aba4fbae218efc9716bcdf7d6b3035331c8` | 64.85 GiB (63.99 GiB in the 14 text shards) |
| [`incoai/Ornith-1.5-122B-DFlash2`](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) | `9baefc1bec85c08193ab442e0ca5f7057669b61d` | 1.7 GiB |

The MLX weights are affine 4-bit in groups of 64, with the router and the shared-expert gate at 8 bits.

## Run

```sh
TF_QWEN27_SLOT_RESUME=0 tensorfold-native serve ~/models/Ornith-1.5-122B-MLX-4bit \
  --drafter ~/models/Ornith-1.5-122B-DFlash2 --name bench --parallel 4 --context 32768 \
  --prompt-cache-gib 0 --no-thinking
```

This is the measured fresh-state configuration: both the prompt cache and slot-0 standing-state reuse are off.
Use `--parallel 1` for a dedicated one-stream server. `--no-drafts` replaces `--drafter` for plain decoding;
`--mtp-drafts 1 --drafter ~/models/Ornith-1.5-122B-mtp` selects MTP instead. MTP and DFlash2 do not combine:
the Metal loader selects one draft source, and in MTP mode `--drafter` names the MTP shards.
The historical [health snapshots](../receipts/ornith-122b-metal/health.txt) used the older build and cache settings,
not this fresh-state configuration.

## Speed

Measured on 13 November 2026: Mac17,15, Apple M5 Ultra, 256 GiB, macOS 27.0 (26A425), Zig 0.17.0,
`tensorfold-native 1.2.7` built with `zig build native -Dcpu=apple_m1` from
`9f7f42296606694443cc04e23a033c556dc459b8`. Binary SHA-256:
`796fe94664b8824b920450037e66d17373ba5ee4f99ed125ccda54b7f7fcbeb4`.
Checkpoint revisions are those above; no weight or arithmetic changes were made for these measurements.

Greedy, seed 1234, 256 tokens, EOS ignored, thinking off, one 16-token warmup, three repetitions.
One stream sends `quicksort-raw`; four streams send the first four entries of `tools/bench_prompts.json`:
`quicksort-raw`, `rust-reverse-words`, `multiply-17-23`, `sleep-tips`. A fresh server was used for each mode/capacity,
with the flags in **Run**. `uptime` was checked before every request; requests waited when one-minute load exceeded
36. Recorded loads in this matrix were 10.8–14.9. The machine was shared: this gate is not GPU isolation.

Rates are medians of three groups (per-stream rates pool their requests); decode excludes first-token latency.
TTFT is client-observed first content, not startup or prefill-only time. Acceptance is summed
`speculative.accepted / speculative.drafted` over timed replies only, excluding warmups and equality controls. The four
replies of a shared round carry that round's counters, so each four-stream group counts once.

| Mode | Streams | Server capacity | tok/s per stream | Aggregate tok/s | First token: median / max | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| plain | 1 | 1 | 92.9 | 92.9 | 0.039 / 0.040 s | — (0/0) |
| plain | 1 | 4 | 82.9 | 82.9 | 0.050 / 0.055 s | — (0/0) |
| plain | 4 | 4 | 48.4 | 191.0 | 0.118 / 0.234 s | — (0/0) |
| **DFlash2** | 1 | 1 | **244.2** | **244.2** | 0.039 / 0.040 s | 65.1% (645/991) |
| **DFlash2** | 1 | 4 | **245.3** | **245.3** | 0.039 / 0.039 s | 66.2% (642/970) |
| **DFlash2** | 4 | 4 | **72.2** | **252.3** | 0.119 / 0.243 s | 72.9% (544/746) |
| MTP, drafting fades¹ | 1 | 1 | 61.4 | 61.4 | 0.056 / 0.080 s | 74.8% (80/107) |
| MTP, not drafting¹ | 1 | 4 | 82.7 | 82.7 | 0.038 / 0.038 s | — (0/0) |
| MTP | 4 | 4 | 58.4 | 213.7 | 0.115 / 0.234 s | 65.8% (388/590) |

¹ Not a measurement of sustained MTP drafting. A lone MTP stream drafts on a server's first requests and then
stops: the one-stream server's three timed replies proposed 107, 0 and 0 drafts (63.3, 50.0 and 61.4 tok/s), the
four-capacity server's lone replies none. Its draft is priced from shared-round costs until it reaches zero and is
never measured again; the fix and its regression test over consecutive served requests are a follow-up engine card.
Sustained one-stream MTP speed, TTFT and acceptance are unmeasured until it lands. The one-stream MTP server also ran
in the same minutes as a plain one-stream run disturbed to 62.0 tok/s, and was not repeated.

DFlash2 is 2.63× plain on dedicated one-stream servers and 1.32× plain in four-stream aggregate throughput, the fastest
mode at both. MTP and DFlash2 do not combine (see **Run**), so there is no joint row.
At four streams, MTP's 213.7 tok/s (repetitions 220.5, 213.7, 212.1) is 12% above plain's 191.0 (185.6, 191.0, 199.0)
and 15% below DFlash2's 252.3 (258.6, 252.3, 248.6). Plain's four-stream aggregate measured 220.6 on the 12–13
November runs of an earlier build with the same plain path; the spread is the shared machine's.

Reproduce each four-capacity cell with the corresponding server mode above:

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:8080 bench \
  --levels 1,4 --tokens 256 --reps 3 --temperatures 0 --seed 1234 \
  --prompts tools/bench_prompts.json --alone --serial --output bench.json
```

For dedicated one-stream cells use `--parallel 1`, `--levels 1`, and omit `--alone --serial`.
For the plain four-capacity run, omit `--serial` as well (its solo controls already use plain decoding).
The benchmark client does not retain draft counters in its JSON: the
[receipts](../receipts/ornith-122b-metal-mtp/README.md) keep each reply's `speculative` object in `*.raw.json`, and
their `check.py` recomputes every row above, the acceptance counts and the lone MTP replies from them.
On the four-capacity servers, all 45 timed replies matched their solo token SHA-256 (15 per mode), and all eight
drafted-mode solo controls matched `draft:false` (four per mode). No timed request failed or was unmeasured.
Every timed 256-token `quicksort-raw` reply in every mode has the same token SHA-256 as the 12 November build's.
The first plain one-stream server measured 62.0 tok/s while the drafted four-capacity servers' `draft:false` controls
ran at 85.0–92.1; the table uses its repeat on a fresh server.

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

The 12 November build's plain and DFlash2 16 × 128 token arrays were identical. On build `ed18f8a08` (this one plus
lone-MTP draft pricing, which changes how many drafts a lone stream proposes, not its tokens), every reply to the 16
prompts carried that plain reply's token SHA-256: DFlash2 one at a time (1,548 of 2,922 drafts accepted), MTP one at
a time (9 replies drafted; 49 of 69 accepted) and MTP in four concurrent groups of four (226 of 337 accepted).
The leading match against the saved mlx-lm reference therefore remains 636/2,048. A teacher-forced scoring pass with
mlx-lm 0.32.2 / MLX 0.32.4 gives mean NLL **0.1126066** for those common native continuations and **0.1186612** for
the reference's own. This is unchanged reference quality, not token identity with mlx-lm.

SHA-256 of the 2,048 token IDs concatenated in fixture order as little-endian uint32:

| Continuations | SHA-256 |
| --- | --- |
| plain, DFlash2, MTP | `af952baa7023863a2ea3b8773bf489080106c76a6dd10af91509731558818a9e` |
| mlx-lm reference | `cbe5b72dd3ee625a508259772f5b812dccd3913ca2dab894bdb83c62f47e1266` |

The public prompt fixture's file SHA-256 is
`b44ae199fd64830124fa224087a389e96823b958d60eafefc22c78e9199c3e18`.
Reference generation is the existing [`mlx_ref.py`](../receipts/ornith-122b-metal/mlx_ref.py). The new scoring pass,
[`score_saved.py`](../receipts/ornith-122b-metal-mtp/score_saved.py), scores the continuations saved in
`quality-plain.json` after the native server stopped, so only one model process occupied the GPU; its per-prompt
rows are `score-nll.json`, and `check.py` recomputes the token equality, MTP counts, hashes and means above.

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
`tools/bench_prompts.json`'s 16 prompts, fed as the engine's test feeds it (the prompt in one pass, then a token a
step, the target and the MTP layer each on its own K/V cache), and the layer's top eight drafts at each step with
their logits. TensorFold's 2,048 drafts equal mlx-lm's best at 1,912, and each of the others is among its top eight,
at most 5.4 logits below its best. The 4-bit MoE target's rows differ between the two kernels' rounding, and that
alone splits mlx-lm too: its stepped drafts and its drafts over one whole causal pass differ at 131 of the 2,048, so
the test does not ask for all 2,048 to be equal.

Plain serving does not load the layer and is unchanged: on fresh `--parallel 1 --no-drafts` servers, alternating
`6d4811726` and this change twice, the 16 reference prompts' greedy 128-token replies are identical (2,048 token ids),
and one stream decodes at 90.8 / 85.2 tok/s before and 88.4 / 85.3 tok/s after (`tools/bench_concurrent.py --levels 1`,
256 tokens, three repetitions).

`serve --mtp-drafts 1 --drafter ~/models/Ornith-1.5-122B-mtp` loads the layer and drafts with it in place of DFlash2:
after every target pass, lone or shared, the layer steps over each drafting stream's kept rows with the tokens the reply
committed after them, and each stream holds its one draft, which the next round verifies beside the pending token. A
shared round is kept when its drafting streams ask for drafts, all at once (a stream that does not draft keeps its one
row), so the layer has stepped every kept chain before it drafts. A refused draft's row is dropped before the layer
steps, so its cache's next write starts at the kept end; prompt chunks step it too. A stream drafts only while the
layer's cache follows every one of its rows from the prompt's start: a resumed or restored prompt drafts nothing.
`zig build test-qwen27-mtp-lanes` (the same two variables) checks that drafted replies are plain decoding's tokens four
streams at once, each drafting and accepting in shared rounds, then three drafting beside a plain one, and then alone
in a reused slot, where the reply keeps 47 of 48 drafts over 96 tokens.

## Not covered

- Resumed or restored MTP prompts, which draft nothing (see **MTP layer**).
- Sustained one-stream MTP: a lone stream stops drafting after a server's first requests (see **Speed**); its speed,
  TTFT and acceptance wait for that engine fix.
- Images, sampled replies and prompts longer than the bench's (under 50 tokens) were not measured here.
