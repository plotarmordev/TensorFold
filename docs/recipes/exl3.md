# EXL3 weights

ExLlamaV3's trellis quantization (EXL3) is what many of the community's best low-bit NVIDIA checkpoints ship as:
`mul1` codebook at 2 to 6 bits for MiMo-V2.6-Flash, Qwen3.8-27B and Flash Next, `mcg` at 4 bits for
GLM-5.3-Flash.

## Where it stands

- The native engine has the format and a reference decoder in core (`zig/src/core/exl3_format.zig`, @akol1, #496),
  and Metal has a standalone EXL3 projection (`zig/src/core/exl3_projection.zig`, @akol1, #681). On CUDA, `qwen3_5`
  serves EXL3 packs of the Qwen3.5 models and of the dense Qwen3.8-27B ([using it](#using-it)).
- CUDA's EXL3 operators, projections and grouped routed experts landed in 1.1.0 (#740, #741, #806, #837). On CUDA,
  GLM-5.3-Flash runs its routed experts through them and `qwen3_5` serves dense EXL3 packs. Another family needs a
  loader for its EXL3 tensors before it can serve a checkpoint.
- The Python engine served EXL3 on CUDA from 0.3.6 to 0.6.6 through the module @vcruz305 wrote (#42): every codebook,
  widths 1 to 8 mixed per tensor, a row-invariant linear for 1 to 128 rows, and grouped routed experts, for the 27B,
  Flash Next and mixed-width packs. It is at the
  [`v0.6.6` tag](https://github.com/ashhart/TensorFold/tree/v0.6.6/src/tensorfold/cuda/exl3).
- This page's format description and arithmetic came with that module. The native decoder and the CUDA operators in
  review follow them.

## Engine versions

The sections below describe the Python 0.6.6 engine's module: its files are under `src/tensorfold/` at the
[`v0.6.6` tag](https://github.com/ashhart/TensorFold/tree/v0.6.6), and its numbers were measured on that engine. The
format, the layer's arithmetic and the traps hold for any implementation.

## What a family has to say

A family whose CUDA engine reads every codebook and width declares it, and TensorFold checks the checkpoint's
`config.json` before downloading anything:

```python
QUANT_METHODS = {"cuda": ("mlx", "exl3")}
EXL3_VARIANT = "any"          # any codebook and any width of BITS (the average bits may be fractional)
```

A family that reads one variant declares it as a dict instead (`families/glm5_next`) and checks it in its own
`check()`. The per-tensor widths are the module's business, not the config's: one checkpoint mixes them across
layers, projections and the experts of one MoE layer, and `config.json`'s `bits` is only the model's average
(`head_bits` is the head's, `mtp_bits` the MTP head's).

## The format, in one paragraph

A quantized linear layer with K inputs and N outputs (both multiples of 128; ExLlamaV3 pads) is a group of
tensors under one prefix: `trellis` int16 `[K/16, N/16, 16 * bits]`, `suh` fp16 `[K]` (or the packed sign words
`su` int16 `[K/16]` of older checkpoints), `svh`/`sv` for the outputs, an optional fp16 `bias`, and a scalar int32
marker tensor `mcg` or `mul1` naming the codebook and storing its magic as int32 (no marker: the original `3inst`). Each tile holds 256 values
in a circular bitstream; the codebook maps a 16-bit state to an fp16 value; the tiles' layout is the tensor-core
permutation. The exact bit arithmetic follows; `tests/test_exl3_format.py` checks `format.py`'s decoder against
one written from this description, bit by bit, for all 27 codebook-and-width combinations.

```text
The tile. A tile holds 256 values in R = 256 * bits bits (16 * bits int16 words). Its int16 words, read in pairs as
little-endian 32-bit words, form a circular bitstream of R bits read from the most significant bit of each 32-bit
word. Value p of the tile (p = 0..255) is decoded from the 16 bits of the stream that end at bit E(p) (exclusive,
wrapping around past the tile's last bit), taken as an unsigned integer s, first bit most significant:

    E(p) = (p + 1) * bits                                   integer bits
    E(p) = ((p + 1) * (2 * KA + 1) - ((p + 1) % 2)) / 2     bits = KA + 1/2: positions alternate KA and KA + 1
                                                            new bits, the odd positions taking the extra bit

and the codebook maps the 16-bit state s to an fp16 value:

    3inst  x = (s * 89226354 + 64248484) mod 2^32; x = (x & 0x8FFF8FFF) ^ 0x3B603B60;
           value = fp16(x & 0xFFFF) + fp16(x >> 16)         one fp16 addition, rounded to nearest even
    mcg    x = s * 0xCBAC1FED mod 2^32, then as 3inst
    mul1   x = s * 0x83DCD12D mod 2^32; h = 1024 + (the sum of x's four bytes);
           value = h * fp16(0x1EEE) + fp16(0xC931)          one fused multiply-add, rounded once (1/147.7, -10.39)

Value p lands in its tile at row 2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1), column l // 4 + 8 * (j >> 2), where
l = p // 8 and j = p % 8: lane l of a warp holds values 8l..8l+7, exactly its B fragments of the two
mma.m16n8k16 of the tile (columns 0-7 and 8-15). The tiles make W_q [K, N], the weight in the rotated domain.

The layer. With H the 128x128 Sylvester Hadamard matrix scaled by 1/sqrt(128), applied to each block of 128 inputs
or outputs,

    y = x @ W + bias,   W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh),
    so  y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias

(``out_scales`` in the config only says whether the quantizer folded per-column scales into svh; the formula is the
same.) Splitting a layer keeps whole tiles and whole Hadamard blocks: by outputs, take columns of tiles and of svh
and all of suh; by inputs, rows of tiles and of suh and all of svh, and add the partial outputs.
```

## Encoding on a Mac

`zig build tf-exl3-encode -Doptimize=ReleaseFast` builds the portable CPU encoder. It needs neither torch nor CUDA:

```sh
zig-out/bin/tf-exl3-encode WEIGHT.npy HESSIAN.npy 2.0 OUT_DIR [--unweighted]
```

`WEIGHT.npy` is the Hugging Face `[N,K]` matrix and `HESSIAN.npy` is its full `[K,K]` calibration Hessian; both
are little-endian C-order FP32 or FP64. Replace `2.0` with `2.5`, `3.0` or `4.0`. The command writes one `mul1`
group as `OUT_DIR/model.safetensors`, reloads it through `core/exl3_load.zig`, and requires the reloaded
`dequantize()` output to equal the encoder's reconstruction bit for bit; that reconstruction is built from the
states the search chose, not by decoding the packed words. Its receipt prints the exact `trace((W-Wq) H (W-Wq)^T)`
for EXL3 and NVFP4 (`core/quant/nvfp4.zig`'s `quantize`: ModelOpt's global scale amax / (6 * 448), E4M3 block-16
scales, E2M1 ties to even), and the encode-only seconds for this one matrix with their 5,900-matrix projection;
writing, reloading and the comparison are not timed. `--unweighted` also prints the error, under the same Hessian,
of the same search run without calibration (`H = I`, every value weighted alike): the same-codebook baseline that
shows what the calibration buys. It doubles the run, so a conversion leaves it off.

The Hessian follows Drama card 32's calibration rows (`drama-exl3-calibration/1`): for expert e, `X_e` is the
layer's `x` rows (the MoE input after `addRmsnorm`) whose `picks` contain e, and gate and up both take
`H_e = X_e^T X_e / n_e`; down takes `A_e^T A_e / n_e`, with `A_e` the SwiGLU of the gate and up projections on
those rows.

Each 16x16 tile is a tail-biting trellis: a 128-wide beam weighted by the rotated Hessian's diagonal runs once over
the tile turned by half to choose the state where the ring closes, then again with the ring's wrapped bits forced to
that state, so every tile closes (closing costs about a tenth over the free beam's error). Tile rows run on every
core. An all-zero output column (Drama's gate and up experts have about a fifth pruned) gets a zero scale and
decodes to exactly zero: any other scale would spread its codebook noise through the output Hadamard into the
whole block.

### Drama 3 experts on this Mac

Layer 0 experts 152, 217 and 242 (353, 321 and 252 calibration rows, the most of any layer-0 expert; 61 of the 256
have none), FP8 E4M3 dequantised with their 128x128 block scales, against card 32's prompt-text rows. Each figure is
`trace((W-Wq) H (W-Wq)^T)` over NVFP4's (ModelOpt scaling, as above), as (expert 152, 217, 242); NVFP4 is about 4.56
bits per weight:

| bits | gate | up | down |
|---|---|---|---|
| 2.0 | 8.74, 6.67, 5.89 | 8.46, 6.26, 5.59 | 9.64, 8.69, 8.34 |
| 2.5 | 4.72, 3.28, 2.94 | 4.43, 3.08, 2.83 | 5.41, 4.84, 4.60 |
| 3.0 | 2.49, 1.63, 1.45 | 2.33, 1.63, 1.33 | 3.08, 2.73, 2.59 |
| 4.0 | 0.73, 0.43, 0.36 | 0.72, 0.43, 0.34 | 1.11, 0.98, 0.93 |

The same matrices' EXL3 error over the same-codebook baseline (`--unweighted`: the same search without calibration,
scored under the same Hessian):

| bits | gate | up | down |
|---|---|---|---|
| 2.0 | 0.926, 0.983, 0.976 | 0.969, 0.943, 0.946 | 0.994, 0.998, 0.992 |
| 2.5 | 0.974, 0.950, 0.963 | 0.948, 0.909, 0.960 | 0.991, 0.996, 0.989 |
| 3.0 | 0.962, 0.984, 0.948 | 0.941, 0.989, 0.932 | 0.997, 0.990, 0.997 |
| 4.0 | 0.993, 0.911, 0.940 | 0.973, 0.916, 0.897 | 0.993, 0.99992, 0.992 |

The calibration wins on all 36 matrices, but by little: 2-10% on gate and up, under 1% on down (expert 217 at 4.0
bits by 0.01%). The search weights only the rotated Hessian's diagonal, with no LDLQ feedback of the
full H, and the Hadamard rotation evens that diagonal out within each 128-row block (a diagonal H becomes a constant
one). With only 252-353 rows each Hessian also has rank at most that, against 2048 or 512 inputs.

The earlier 0.82-0.90x NVFP4 figure was K4 EXL3, and at 4.0 bits this encoder reaches 0.34-0.73x on gate and up
and 0.93-1.11x on down. At 2.0-3.0 bits it misses that bar because it stores 1.5-2.6 fewer bits per weight than
NVFP4 (0.45-0.66x its bytes), and each bit removed costs roughly 4x the error.

Encode-only seconds (the mean of the three experts) on an Apple M5 Ultra (12 performance and 24 efficiency cores,
256 GB, macOS 27.0, Zig 0.17.0 ReleaseFast, all 36 cores), and the projection for 5,900 experts. The Mac was shared:
its load average stayed between 57 and 85 throughout, and an earlier, quieter run took 16.1, 20.2, 25.6 and 38.4
seconds per expert, so these are upper bounds:

| bits | gate | up | down | seconds per expert | 5,900 experts |
|---|---|---|---|---|---|
| 2.0 | 10.4 | 9.1 | 8.2 | 27.8 | 45.5 h |
| 2.5 | 13.1 | 12.9 | 14.6 | 40.5 | 66.4 h |
| 3.0 | 20.1 | 19.1 | 19.3 | 58.5 | 95.9 h |
| 4.0 | 32.0 | 32.1 | 26.9 | 91.1 | 149.3 h |

All of 2.0, 2.5 and 3.0 together take 101 to 208 hours (4 to 9 days) by that range: slow but within reach on the
CPU, so a Metal path is left to its own card if the conversion needs it. This is a CPU reference converter, not a
claim of CUDA quantizer quality or throughput.

## Using it

```python
from tensorfold.cuda.exl3 import format as exl3_format
from tensorfold.cuda.exl3.linear import Exl3Linear

layer = Exl3Linear.load(model_dir, "model.language_model.layers.0.self_attn.q_proj")
y = layer(x)                                  # x [1..128, K] fp16/bf16/fp32 -> y, same dtype
```

`load` reads one group with safetensors and copies the trellis, never changing it, into the read order the
kernel wants (`layout="strips"`: a 128-column block's tiles in k order, so a warp walks contiguous words).
`layout="stored"` reads the checkpoint's own order instead; both give the same bits.

To serve a whole checkpoint, walk `format.scan(model_dir).groups`, build one `Exl3Linear` per group, and treat
the checkpoint's plain tensors (`checkpoint.plain`: embeddings, norms, routers, sometimes a head) as ordinary
weights of the family's engine. `python -m tensorfold.cuda.exl3.inspect MODEL_DIR` prints exactly that split,
the bits per category, and every group that does not parse.

## The layer's arithmetic

```
y = ((((x * suh) @ H_K) @ W_q) @ H_N) * svh + bias        H = the 128x128 Hadamard / sqrt(128)
```

`rot_in` rotates the input once per call (fp16 out, as ExLlamaV3 does), `linear` decodes each k tile straight
into tensor-core fragments and multiplies in fp32. Rows are independent by construction: the k ranges of warps
and of K splits depend only on (K, N) (`plan(k, n)`), every sum runs in a fixed order, and `mma.m16n8k16` keeps
its rows independent, which is the verify path's contract (`docs/recipes/cuda.md`).

## Prompts

Prompt chunks take their own arithmetic, as the MLX 4-bit path's prompt matmul does (`src/tensorfold/cuda/exl3/prefill.py`): the
input rotation is decode's, W_q is decoded once a chunk into fp16, a fixed-tile fp16 GEMM with fp32 accumulation
multiplies it, and its epilogue rotates each 128-column block (the accumulator's bf16 high and low halves times
H) before `svh` and the bias. Tiles depend on the shape alone, so a row's bits never depend on its chunk and a
resumed prompt equals a fresh one; they differ from decode's, so the engines keep prompt ends and prefill a reply
again, as for the MLX checkpoints. A family's head, read one row a prompt, keeps the decode linear.

## Numbers

`tools/bench_exl3.py` prints this table. It cycles over enough weight copies to exceed L2, so these are DRAM
figures, medians of 9 runs of a CUDA graph of back-to-back calls on an idle DGX Spark (GB10), the same tensors
through ExLlamaV3's own `LinearEXL3`:

| MiMo-V2.6-Flash 2.50bpw tensor | bits | bytes | rows | TensorFold | ExLlamaV3 |
| --- | --- | --- | --- | --- | --- |
| `layers.0.self_attn.q_proj` | 2 | 12.6 MB | 1 | 57.0 us (221 GB/s) | 59.0 us (213 GB/s) |
| `layers.0.self_attn.q_proj` | 2 | 12.6 MB | 16 | 60.0 us (210 GB/s) | 65.9 us (191 GB/s) |
| `layers.2.self_attn.o_proj` | 3 | 12.6 MB | 1 | 72.9 us (173 GB/s) | 73.0 us (172 GB/s) |
| `layers.2.self_attn.o_proj` | 3 | 12.6 MB | 16 | 68.4 us (184 GB/s) | 63.5 us (198 GB/s) |
| `layers.41.self_attn.o_proj` | 6 | 25.2 MB | 1 | 114.0 us (221 GB/s) | 113.5 us (222 GB/s) |
| `layers.41.self_attn.o_proj` | 6 | 25.2 MB | 16 | 123.7 us (203 GB/s) | 116.0 us (217 GB/s) |
| `lm_head` | 6 | 468.7 MB | 1 | 2074.5 us (226 GB/s) | 2071.8 us (226 GB/s) |
| `lm_head` | 6 | 468.7 MB | 16 | 2113.0 us (222 GB/s) | 2045.8 us (229 GB/s) |

Rows cost little: every tensor above is within 5% from 1 row to 16, because one program's weights are decoded
once and reused by all its rows. 6-bit is at ExLlamaV3's level (221-226 GB/s of a 237 GB/s streaming ceiling on
this box) and so is 2-bit at 1 row (221 vs 213 GB/s). 2 bits is where the decode itself is the work (four weights
per byte), so the loader there has a warp fetch a k step's words as one coalesced run and hand each lane its two
words of a 16x16 tile by shuffle, and keeps the next k step in flight while the current one decodes; that is worth
40% at 2 bits (q_proj 156 -> 221 GB/s at 1 row, o_proj 76 -> 187) and 10-15% at 3 to 5 bits. Above 6 bits a step
is 224 or 256 B of words a warp and prefetching it measured 1-2% slower than loading per tile, so those widths
load per tile. What is left is the small-N 3-bit shapes at 16 rows (o_proj 184 vs 198 GB/s) and, on the 27B packs,
rows 8 and 16 of the 3-bit shapes, where the split a shape alone picks is 3-19% off the best cell of the
(K split, warps) grid. An earlier version of this kernel streamed the tiles through shared memory with `cp.async`; that was
three to five times slower than these per-lane cached loads on GB10, whose L1 serves a warp's 256-byte search
from one line.

## Traps

- `config.json`'s `bits` is the average, `head_bits` the head's. Read the width from each tensor's trellis
  (`16 * bits` words per tile), never from the config.
- Half bits (1.5, 2.5, 3.5) exist only with the `mul1` codebook, and their tiles are not a whole number of
  bits per value: the bitstream alternates KA and KA + 1 bits, so the window ends are `E(p)` of `format.py`,
  not `(p + 1) * bits`.
- `suh`/`svh` are scales, not signs: the sign of a weight lives in its 16-bit state. The packed `su`/`sv` words
  of older checkpoints are signs only (`1` bit set = -1), and `format.unpack_signs` expands them.
- A model may keep some tensors unquantized (a head, an MTP head, embeddings, norms): `scan` lists them under
  `plain`.
- The Hadamard blocks run along K and N, so both must be multiples of 128. A split of K must keep whole tiles
  and whole blocks: `plan` only ever splits 128-aligned k ranges.

### Consolidated Flash Next n-grams

Flash Next EXL3 packs may store `ngram_embedding.trellis` as one int16 `[rows, words]` tensor instead of
`ngram_embedding.shard_N.trellis`. Both layouts use the same scale-plus-160-values row codec. CUDA maps either
layout read-only and gathers only requested rows; the consolidated layout does not require re-sharding or
requantization. Admission treats the packed table as reclaimable mapped pages, while head metadata remains
resident. Mapped pages still consume physical RAM when touched; keep the normal cache/workspace reserve.

A backend-neutral Zig reference decoder for this format (@akol1, #496) lives at
`zig/src/core/exl3_format.zig`: the bitstream, the three codebooks, tile placement and
the Hadamard rotations, written deliberately naive (one bit at a time) so it reads
exactly like this page. It is the correctness oracle for the native tile decoders: the
standalone Metal projection (`zig/src/core/exl3_projection.zig`, @akol1, #681) today, and
a CUDA one to come; `zig test zig/src/core/exl3_format.zig` from the repository root
reproduces the numpy oracle on every codebook and width and on real checkpoint bytes
(the oracle vectors ship beside the test). Its real checkpoint bytes are one 128x128 block from
`UnstableLlama/Qwen3.5-2B-exl3-8.00bpw` (mcg codebook, 8.00 bpw), in `zig/tests/fixtures/exl3_oracle.bin`.

The scalar decoder and its oracle tests do not register an EXL3 model on their own; the native CUDA kernels below
are checked against them.

## Native CUDA (Zig)

The Zig engine carries the same arithmetic as hand-written CUDA, ported from the Python engine's kernels so a
row's bits match it:

- `zig/kernels/cuda/exl3_decode.cuh`, `exl3_linear.cu`, `zig/src/cuda/exl3.zig`: the input rotation and the linear
  for 1 to 128 rows, every codebook and width (the `mul1` half steps included), with the K split planned from the
  layer's shape alone. `exl3.Dense` runs prompt rows in 128-row blocks of the same kernel, so a prompt row's bits
  equal its decode-row bits wherever its chunk starts; `qlinear.Weight` has an `.exl3` arm for both.
  The split counters are zeroed on the calls' stream when they grow, and each block's last program resets its own.
  On sm_90 and newer the input rotations and the linear, alone or grouped, go out as programmatic dependent
  launches: a linear loads its weights while the kernel before it finishes, then waits for its rows.
  `TF_EXL3_PDL=0` makes them plain stream-ordered launches with the same bits, to time a change without them or to
  rule them out when chasing a fault; below sm_90 they are plain already.
- `zig/kernels/cuda/exl3_experts.cu`, `zig/src/cuda/exl3_experts.zig`: grouped routed experts (member lists,
  each expert's rotation, gate and up, GLM's SwiGLU roundings, down and the weighted combine), widths mixed within
  a layer; an expert's member count never changes a row's bits.
- `zig/src/core/exl3_load.zig`: every trellis group of a checkpoint from its safetensors headers;
  `zig/src/cuda/exl3_upload.zig` puts a group, or a layer's experts, on the device.

`tf-cuda-test exl3 DIR` checks them: every width and codebook against float64 references built on
`exl3_format.zig`, row and chunk invariance, and, when `DIR` holds bytes the Python engine recorded on real checkpoint
layers, those layers bit for bit, two whole layers of RED-SNOW-5.3-Flash (a KDA and a DSA layer, all 288 experts
under real routing) included. The program that records them runs the Python engine and is not in this repository;
`none` skips those checks. `tf-cuda-test exl3-tp2 RANK HOST:PORT DIR` runs the GLM split rules on two ranks over
NCCL (output-column cuts, input-row cuts with fp32 partials summed rank 0 first, and the experts' intermediate cut)
and checks every rank's bytes and the sums the same way.

The Metal projection (#681) and these CUDA pieces are what a family's engine takes, through `qlinear` and
`exl3_experts`. On CUDA, `qwen3_5` serves EXL3 packs of the Qwen3.5 MoE models and of the dense
[Qwen3.8-27B](qwen3.8-27b.md#cuda), whose DFlash2 drafter reads the head through column views
(`exl3.Weight.columns`: whole 128-column blocks with the parent's strides).
