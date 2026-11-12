# MiniMax-H3-Image's transformer on Metal (M5 tensor units)

MiniMax-H3-Image makes pictures from text, and edits pictures from instructions. Its transformer is MiniMax H3's,
with the audio layers removed: 50 blocks, hidden 5,376, 56 heads of 128, an FFN of 14,336. It is distilled to six
sampling steps with no guidance, and each step has its own output head in place of the transformer's.

`tf-h3-image` runs that transformer on Metal from a picture case, using the H3 family's own blocks
([minimax-h3-metal.md](minimax-h3-metal.md)). It is a denoiser beside the engine, not a token family. Nothing in the
lane engine, the server or another family changes, and a clip's case runs exactly as before.

## Text to image, end to end

Three tools, each on Metal. The prompt and decoder run on any Apple GPU; the transformer needs an M5's tensor units:

```sh
zig build tf-h3-image-prompt tf-h3-image tf-h3-image-decoder -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42
zig-out/bin/tf-h3-image MODEL_DIR out/case.safetensors out/case free
zig-out/bin/tf-h3-image-decoder MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
```

1. With a size and a seed, `tf-h3-image-prompt` writes a whole text-to-image case:
   - the prompt's rows;
   - the canvas, grown to 2048 x 2048 at the size's ratio when smaller, with sides snapped to 32;
   - the requested size;
   - the starting rows: the seed's first draw of torch's CPU `randn` over the latent [24, height / 16, width / 16],
     made natively (`moss_image/noise.zig`) and patchified.
2. `tf-h3-image ... free` runs the six steps from those rows. It writes the latent, and exits 0 with nothing to gate.
3. `tf-h3-image-decoder` decodes the latent. Its rows and columns are the canvas over 16. It resizes the picture back
   to the requested size when that differs from the canvas.

### Resident: one process, every picture warm

`tf-h3-image-server` loads the prompt encoder, the transformer's weights and the decoder once, then makes one picture
per stdin line, writing the same files as the three tools above:

```sh
zig build tf-h3-image-server -Dcpu=apple_m1 -Doptimize=ReleaseFast
printf '2048x2048 42 out/a prompt.txt\n2048x2048 7 out/b prompt.txt\n' |
  zig-out/bin/tf-h3-image-server TEXT_ENCODER_DIR MODEL_DIR
```

A request line is `WIDTHxHEIGHT SEED OUT_PREFIX prompt.txt`. This server accepts text-to-image requests only, not
reference images or editing cases. The transformer's kernels and buffers depend on the picture's layout (canvas,
rows and text tags), so a request with the last one's layout keeps them and only replaces the prompt's text rows
(`Model.retext`); another layout releases them and binds anew over the same mapped weights. `Binding.fits` compares
the canvas, row count, reference count and tags; it assumes there are no reference rows, so those determine all
rotary positions. Supporting edits would also require checking reference geometry and refreshing reference
conditioning, not just comparing row counts.
Each request prints one line of stage seconds: prompt, case, transformer (`reused` or `bound`), 6 steps, decoder, total.
A bad request prints its error and the server reads the next line.

At startup the server puts every weight file it maps (text encoder, refiner, transformer, step heads) and the decoder's
bf16 weights in one residency set on the three stages' queues (`holdWeights`), and prints the GiB it holds: 117.6 on
the release folder, the refiner's shard counted twice since it is mapped twice. macOS unwires GPU memory idle for a
second or two, so without the set each stage's first commit re-wires its weights on every request. Weights past 70% of
RAM, or a macOS without residency sets, leave the server running as before. There is no keepalive: a request after a
pause of more than a second or two re-wires once, in its prompt (about 1.4 s more, measured after 20 and 60 s).

Run `zig build test-h3-image-server -Doptimize=Debug` or `-Doptimize=ReleaseSafe` for the synthetic resident gates.
Both pass all four tests: fresh/reused latent parity across prompt and layout changes, failed-bind cleanup including
the final tag allocation, over-long-line recovery, and the shared serializer's scratch cleanup. Picture fixtures use
the aligned safetensors serializer so these checks run with alignment safety enabled. The line-reader gate checks
that the next request is read, not full image generation after an oversized line.

## Edit, end to end

An edit adds two tools before the prompt: the edit encoder makes each reference's latent rows, and the vision tool
makes its rows for the text encoder. The transformer needs an M5; the other tools run on any Apple GPU:

```sh
zig build tf-h3-image-encoder tf-h3-image-vision tf-h3-image-prompt tf-h3-image tf-h3-image-decoder -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-encoder MODEL_DIR reference.png 42 out/ref 2048x2048
zig-out/bin/tf-h3-image-vision TEXT_ENCODER_DIR MODEL_DIR reference.png out/ref.vision.safetensors
zig-out/bin/tf-h3-image-prompt TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/case.safetensors 2048x2048 42 \
  --reference out/ref.vision.safetensors --anchors out/ref
zig-out/bin/tf-h3-image MODEL_DIR out/case.safetensors out/case free
zig-out/bin/tf-h3-image-decoder MODEL_DIR out/case.latent.f32 128 128 out/case.png 2048x2048
```

1. `tf-h3-image-encoder` makes the reference's clean and anchor rows ([Edit encoder](#edit-encoder)). Give it the
   request's seed and its canvas, which the prompt tool prints (a 2048 x 2048 request's canvas is itself). It draws the
   target's noise and then the anchor's, as the edit's one generator does.
2. `tf-h3-image-vision` makes the reference's vision and DeepStack rows ([Vision tower](#vision-tower)).
3. `tf-h3-image-prompt --reference` puts `Picture 1: ` and the reference's vision block before the prompt, runs the
   text encoder with the reference's rows, and tags the block's rows 0. `--anchors` writes the encoder's grid and
   anchor rows as the case's `references` and `anchors`, beside the canvas and the target's starting rows.
4. `tf-h3-image` steps only the target rows, holding the reference's at 0.999. The decoder runs as for text to image.

`--reference` may be given more than once, in upload order, each with its own `--anchors`. The encoder draws each
reference's noise as a one-reference edit does, so only a one-reference edit draws exactly the reference pipeline's
noise.

## What runs here, and what the case brings

Made natively, from the request and the model folder:

- **The canvas.** A request below 2048 x 2048 is drawn at the same aspect ratio on a canvas of about that area. Every
  side is snapped to 32 pixels, rounding half to even, as Python's `round` does. One transformer row covers 32 x 32
  pixels: the VAE's 16, then a 2 x 2 patch.
- **The schedule.** `pdd_config.json` holds the step count and its steps + 1 sigmas, which fall strictly from 1 to 0.
  Each sigma goes through a float32 timestep (u = f32(1 - sigma), then sigma' = 1 - u), so the timestep the model sees
  plus its sigma is exactly 1. Each step moves the target rows by plain Euler, x + (sigma'_k - sigma'_k+1) v, in
  float32. That is the form the heads were distilled with, not the clip tool's ratio form. The tool refuses a run with
  any other step count.
- **The step heads.** `pdd_fused_heads.safetensors` holds six float32 heads, [6, 96, 5,376] and [6, 96]. Step k's
  head replaces the final layer's `video_out`. A picture's case may hold such per-step heads; a clip's case may not.
- **The rows' positions.** Each row gets a (t, h, w) position:
  - text row i sits at (i, 0, 0);
  - reference k sits at t = text + k;
  - the target sits at t = text + the reference count.

  A block's h and w are centred on 16 and normalised by the block's own area, in float64, then rounded once to
  float32. A square block spans [0, 32): a 2048-pixel target in steps of 0.5, a 1024-pixel reference in steps of 1.
- **The rotary table.** Each axis has 16 frequencies, read from the checkpoint's `rope.inv_freq`. That gives 48 angles
  a row, whose cosines and sines rotate the first 96 of each head's 128 dims. The heads kernel rounds them to bf16, as
  for clips.
- **The tables.** A row's timestep is the step's, except references, which are held at 0.999 for every step. Its
  AdaLN line is that timestep's index times three plus its modality (text 1, picture 0). The AdaLN tables come from
  the checkpoint as for the released MiniMax-H3 clips (`modulation.zig`).
- **Attention.** Dense over every row, in tiles of 64: the prefix (text and references), then the target.

The picture case brings the rest. The prompt tool writes `canvas`, `text`, `text_tags`, `noise`, and with an edit's
`--anchors` `references` and `anchors`; a reference pipeline's case adds its own rows for the gates:

| Tensor | Holds |
|---|---|
| `canvas` | I32 [2]: the canvas's width and height |
| `text` | BF16 [text, 5,376]: the prompt's rows as they enter block 0 (after `condition_proj` and the two-block token refiner) |
| `text_tags` | I32 [text]: 1 for text tokens, 0 for the tokens of a reference picture's vision block (all 1 when absent) |
| `references`, `anchors` | An edit's reference grids, I32 [n, 2] in latent cells (rows, columns), and their anchor rows, F32 [rows, 96] |
| `noise` | F32 [target rows, 96]: the starting target rows |
| `step_in`, `velocity` | F32 [6, target rows, 96]: the reference's input and velocity at each step |
| `latent` | F32 [target rows, 96]: the reference's rows after the sixth step |
| `positions`, `rotary_cos`, `rotary_sin`, `timesteps`, `deltas` | The reference's own, checked against ours |

Rows are 96 wide: 24 latent channels times a 2 x 2 patch, channel-major, then the patch's row, then its column.
Patches run in row-major order. The case must be written aligned (Python's `safetensors` writer does this).

## Prompt

`tf-h3-image-prompt` makes a picture case's `text` and `text_tags` from a prompt. It runs on any Apple GPU: no tensor
units are needed.

```sh
zig build tf-h3-image-prompt -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-prompt TEXT_ENCODER_DIR MODEL_DIR prompt.txt out/prompt.safetensors
```

`TEXT_ENCODER_DIR` is MiniMax-H3's `text_encoder/` (Qwen3-VL). `MODEL_DIR` is the MiniMax-H3-Image folder, for its
`tokenizer/` and `transformer/`. The prompt is used verbatim, except for one trailing line ending: no chat template and
no system prompt.

1. **Ids.** The prompt is tokenised on its own, with no special tokens added, and cut to its first 4,096 tokens.
   Every row is tagged 1 (text).
2. **The text encoder.** Qwen3-VL's text model runs 50 of its 64 layers:
   - hidden 5,120, with 64 query and 8 KV heads of 128;
   - q and k RMSNorm, and a SwiGLU of 25,600;
   - causal attention, with rotary at theta 5,000,000.

   Text positions count up, equal on M-RoPE's three axes. The rows leaving layer index 49 are kept unnormalised. The
   final norm and layers 50 to 63 are never mapped or run.
3. **The front end.** `condition_proj` and its bias map each row to 5,376. The two token-refiner blocks then run over
   every row as one document, with no rotary and no AdaLN, and `token_refiner.final_norm` ends it.

It writes `text` BF16 [rows, 5,376], `text_tags` I32 [rows] and `ids` I32 [rows], aligned. `tf-h3-image` reads the
first two as a case's own.

**Precision.** Weights are bf16, read in place. Products run in simdgroup matrices with fp32 sums. Norms keep fp32
statistics, and the softmax and its sums stay in fp32. Each step rounds to bf16 where the reference's bf16 arithmetic
does:

- the text model's norms round x r before the weight;
- its rotary turns by bf16 cos and sin;
- its SiLU rounds once;
- the refiner rounds as H3's blocks do: x r w at once, then the sigmoid, then the product.

**Edits.** Each `--reference` file (from `tf-h3-image-vision`) adds its label, `Picture i: `, and its vision block
before the prompt: vision start, one image pad per token, vision end, all tagged 0. The encoder places the reference's
rows at its image pads and adds its DeepStack rows there after layers 0, 1 and 2. M-RoPE gives the pads (t, h, w) over
the merged grid, and the text after a block resumes past its larger side.

**Gates, fixed before the first run except the whole-run gate below.** `tf-h3-image-prompt selftest [WORK_DIR]`
writes a tiny synthetic checkpoint and loads it as a real run does. The checkpoint is a 256-wide text model with 4
query and 2 KV heads, three layers written and two run, and a 384-wide refiner of 2 heads. The selftest checks 70 rows
against a float64 host reference with the same bf16 rounding points:

- **Each kernel alone:** at least 99.9% of values bit-equal, none further than 1/32 of its row's RMS. Kernels: the
  product, both norms, heads with rotary, fused heads, causal and full attention, both SiLU gates, the residual add.
- **Whole runs:** a text prompt; a reference's vision and DeepStack rows at M-RoPE positions, read from a file as
  `--reference` reads one; the refiner on the GPU's encoder rows. Each must be within twice a float32-sum host run's
  distance from float64, in both 1 - cosine and relative error, with cosine at least 0.999. This gate replaced a fixed
  cosine bound of 0.99999, which the refiner missed at 0.9999884 on a second seed while every kernel alone was
  bit-exact.
- **A second run:** byte for byte the same.

On an M3 Ultra, every kernel alone is 99.97% to 100% bit-equal. The whole runs sit 0.7 to 1.03 times as far from
float64 as the float32-sum run.

`zig build test-h3-image-prompt` runs the prompt rules without a GPU:

- ids per segment, the 4,096 cut, tags, pad rows and vision token counts;
- M-RoPE positions and the interleaved rotary table;
- the Qwen copy's size and LANCZOS pixels;
- the encoder's and refiner's configs, and the text model's tensor prefix.

`zig build compile-h3-image-prompt` compiles the tool without running a shader.

## Run a case

```sh
zig build tf-h3-image -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image MODEL_DIR picture.safetensors out/case [once | free]
```

`MODEL_DIR` is the MiniMax-H3-Image folder: `model_index.json` (the image pipeline's class and partition),
`transformer/` (its config must say `audio_enabled: false`), `pdd_config.json` and the heads file. The tool:

1. checks the case's positions, rotary table, timesteps and Euler steps against ours;
2. runs each step on the reference's own input rows (teacher-forced) and compares the target's velocity;
3. runs all six steps from the noise with our own rows.

It writes `out/case.latent.f32`, the normalised latent [24, rows, columns] in float32, ready for the reference's
decoder. `once` runs the first teacher-forced step alone; `free` runs only the free run. It needs the M5's tensor
units, as the clip tool does.

It and `tf-h3-image-server` make way for live speech on the same GPU. The speech server holds a shared flock on
`/tmp/tensorfold-speech.lock` while it synthesizes (`zig/src/server/gpu_turn.zig`). The tools submit each block as its
own command buffer, with three queued, and wait before the forward, between blocks, after profile stages and after the
step's output while that lock is held. During an earlier 4096 x 4096 render, five streamed Drama 3 voice notes reached
first audio in 1.3 to 1.6 s, against 9.5 to 10.8 s without yielding. The run resumes on the same rows with a
byte-identical latent.
Each step and profile prints `timing_valid`: benchmarks must exclude every `false` sample, including zero-wait
samples. A separate 16-byte generation in `/tmp/tensorfold-speech.lock.stamp` detects speech that starts and ends
between lock polls. Missing, malformed or inaccessible signals invalidate timings; uncoordinated forwards do too.
Both processes must use this protocol; an old speech binary cannot certify clean timings. Speech remains best-effort
on filesystem errors: if both publishing and invalidating a generation fail, a short request can remain unobserved.
Speech retries its shared hold for up to 50 ms past a render's microsecond exclusive probe instead of losing it, and
never waits for GPU work. The prompt encoder (`tf-h3-image-prompt`, the server's first stage) waits before each pass
with three queued, and the decoder (`tf-h3-image-decoder`, its last) before each batch of tiles; both print the seconds
they waited.

### Speech contention receipt, 12 November 2026

Apple M5 Ultra, 256 GiB RAM, macOS 27.0 (26A425), Zig 0.17.0, ReleaseFast with `-Dcpu=apple_m1`.
The branch's isolated Drama 3 NVFP4 server used the consented Capri catalog, CFG 1.5 with paired submission,
one lane and context 8192. The live release server was not changed. More than 170 GiB was free before loading.
The client timed from sending each HTTP request until reading the first two PCM bytes, not response headers.

The reference cache was warmed before starting `tf-h3-image MODEL_DIR CASE OUT free`. The case was card 35's
4096-square, seed-42, 24-text-row case (SHA-256 `af4f85f19ebeb4e4af5beada167d8e8fd00bf1b880541b88a60c0a2b5b6dd66b`).
Five sequential requests, seeds 42–46, used `stream=true` and `response_format="pcm"`:

| Public input | First PCM (s) | PCM bytes | Status |
|---|---:|---:|---:|
| The little boat crossed the harbor before the sun came up. | 1.168 | 212992 | 200 |
| A blue bicycle waited beside the garden gate. | 1.174 | 180224 | 200 |
| Please put the fresh apples in the bowl on the table. | 1.208 | 196608 | 200 |
| We can take the quiet path through the woods today. | 1.265 | 180224 | 200 |
| The train will arrive at the station in a few minutes. | 1.357 | 192512 | 200 |

The render was running at the start and end of every request. Step 1 waited 12.68 s and reported
`timing_valid=false`; steps 2–6 waited zero and reported `true`. The render exited 0 and its final rows matched
the uncoordinated baseline byte-for-byte (SHA-256 `b3726693636bbe7ce77eac8da41868bd959262646bfc5ac889e2c939ee1cd5fd`).
The server drained and stopped afterward. This is a transformer-step benchmark exclusion check, not a clip-profile
receipt: a valid audio/video clip case is still required to exercise `tf-h3-dit` directly.

Cold reference conditioning remains outside the demonstrated latency bound: first PCM without a render took
14.819 s while building 1284 cached reference rows, versus 0.528 s on the next idle request. A cold request during
a render took 15.332 s. The five sub-three-second results therefore establish warm live-voice contention relief,
not a cold-start guarantee. Generation checks now bracket the complete wall interval, including output copies.

### Every stage of the resident server, 13 November 2026

Same machine, speech server and client, from this branch on port 18899; `tf-h3-image-server` drew card 34's prompt at
4096x4096, seeds 42 and 43. The server's first request after start, with its reference rows still unbuilt in memory,
reached first PCM in 3.019 s idle; the next took 0.429 s. During the render (first PCM, all HTTP 200):

| Where the notes landed | First PCM (s) | The render's own record |
|---|---|---|
| Denoising, one note every 17.5 s | 0.806, 1.208, 1.179, 1.204, 1.147 | five steps waited 2.37-2.76 s each, all `timing_valid=false`; one clean step `true` |
| Back to back over step 6 | 0.412-1.131 (twelve notes) | step 6 waited 23.14 s, `false`; steps 1-5 `true` |
| Back to back over the decoder | 1.030, 0.517, 0.590, 0.517 | 441 tiles waited 9.30 s; all six steps `true` |

Each run's seed-42 rows matched the 12 November baseline byte-for-byte (`b3726693...`). The prompt encoder runs
0.3 s a request at this size, too short to target from the client; `test-h3-image-server` covers it with a held turn.
The isolated server was stopped afterward; the live release server was not changed.

## Gates, fixed before any run

These are the bounds `tf-h3-image` checks (`picture.gates`). It exits 1 when one fails, and 2 when the case holds no
reference to gate against.

- Positions, timesteps and Euler steps: bit-identical to the reference's.
- The rotary table: at most 1 bf16 ULP from the reference's.
- Each teacher-forced step: target velocity cosine at least 0.999 and relative error at most 0.03.
- The free run's final latent: cosine at least 0.99 against the reference's.

Attention here is dense, so no routing floor applies. FastH3's 0.990 at step 1 came from its top-k tiles.

## Decoder

`tf-h3-image-decoder` turns `tf-h3-image`'s latent into a picture on Metal. It runs the VAE's 2D ViT decoder (24 blocks
of 1,024, 16 heads of 64, 4 register tokens) on any Apple GPU. Its kernels compile from source when it loads and use no
tensor units.

**What runs:**

1. The latent file (normalised, [24, rows, columns] float32) is mapped back with `latents_mean` and `latents_std` in
   float32.
2. It is cut into tiles of 16 x 16 cells (256 pixels) overlapping by at least 64 pixels, as `picture_vae.zig` splits
   it.
3. Each tile runs `post_quant_conv`, then its cells become tokens through `x_embedder`, with the 4 registers and one
   zero token appended.
4. The 2D rotary table is built, and the 24 blocks run, then `norm_out` and `proj_out`.
5. Each token's 768 values are unpatchified to its 16 x 16 pixels. Up to 16 tiles run in one command buffer.
6. Tiles are blended in float32: the top band against the tile above, then the left band against the tile to the left,
   both as decoded.
7. The result is mapped to [0, 1] with ImageNet's statistics, clamped and truncated to bytes.
8. If the request's size differs from the canvas, the picture is resized with Pillow's LANCZOS. Then it is written as
   a PNG.

**Precision:**

- Every parameter is rounded from the file's F32 to bf16 when loaded, and the latent is rounded to bf16 at
  `post_quant_conv`.
- Products are bf16 with float32 sums, the bias added before the one rounding.
- RMSNorm's statistics and weight are float32, rounded once; q and k are RMS-normed with no weight.
- The rotary ids and 2π·id are in bf16 arithmetic, the angles in float32, cos and sin in bf16.
- Rotation, LayerScale, the residuals and SiLU are bf16 elementwise.
- Attention takes bf16 q, k and v, with float32 scores, softmax, probabilities and sums. The reference's fused
  attention may round the probabilities to bf16 before the value product. A reference case will say which.
- `norm_out` is float32; `proj_out` takes bf16 in and gives bf16 out, held in float32. The blend is float32.

```sh
zig build tf-h3-image-decoder -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-decoder MODEL_DIR out/case.latent.f32 ROWS COLUMNS out/case.png [WIDTHxHEIGHT]
zig-out/bin/tf-h3-image-decoder selftest
```

`ROWS` and `COLUMNS` are the latent's grid in cells: the canvas's height and width over 16, so a 2048 x 2048 canvas is
`128 128`. `WIDTHxHEIGHT` is the requested size.

**Selftest.** `selftest` runs the kernels on synthetic weights against a float64 host reference with the same
rounding points, and exits 1 past a bound. On an M3 Ultra:

| Check | Result |
|---|---|
| The embedding and `proj_out` | within 1 ULP |
| Each block, on the GPU's own input | 93-96% identical, worst 4 ULPs |
| The output after two chained blocks | relative error 3e-3 |
| The output after 24 chained blocks | relative error 1.5e-2 |
| Decodes in different batch sizes | bit-identical |
| The picture's bytes | at most 1 apart |

The host's own float32 sums spread as far from its float64 sums as the GPU's do.

All 24 blocks take 13.4 ms a 16 x 16 tile, so about 1.6 s of GPU time for a 2048 x 2048 canvas (121 tiles).
`zig build test-h3-geometry` covers the decoder's host rules. `zig build compile-h3-image-decoder-host` compiles the
tool without the Metal compiler.

## Edit encoder

An edit's reference reaches the transformer as clean latent rows made by the VAE's CNN encoder. `tf-h3-image-encoder`
makes them on Metal in fp32, on any Apple GPU; no tensor units are needed. Its kernels are compiled from source when it
starts, so it builds without the Metal compiler.

- **The VAE copy.** The reference is read upright in RGB, with any alpha dropped. A short edge already in 1024..2048
  keeps the original pixels. Otherwise the picture is scaled to the nearer bound, each side snapped to 32 (halves to
  even) and resampled with Pillow's LANCZOS: 512x768 becomes 1024x1536, 3000x4000 becomes 2048x2720, and 1536x2048 is
  kept. A side under 32 pixels, or a copy over 64 MP, is refused.
- **The input.** Bytes / 255, then ImageNet's mean and std, in fp32, cropped top-left to multiples of 16 pixels.
- **The network.**
  - Each 3x3x3 causal convolution runs as the 2D convolution of its last time slice, reflect-padded by one.
  - `conv_in` maps 3 channels to 128.
  - Six levels of two ResNet blocks follow, at widths 128, 256, 256, 512, 512 and 1,024, with `nin_shortcut` where the
    width changes.
  - Levels 0 to 3 end with a stride-2 downsample that reflects the right and bottom edges by one pixel.
  - Then GroupNorm(32, eps 1e-6) with SiLU, `conv_out` (1,024 to 48) and `quant_conv`.
  - Every tensor name and shape is checked against the checkpoint's header before anything loads.
- **Tiles.** 256-pixel tiles overlapping by at least 64, split as the decoder splits. Each tile's 48 moment channels
  are blended in latent cells (overlap / 16) with the decoder's rule.
- **The rows.**
  - The posterior sample, mean + exp(0.5 clamp(logvar, -30, 20)) x noise, is rounded to fp16.
  - It is normalised with the latent statistics and cropped top-left to the even grid: 1024x1024 gives 64 x 64, and
    1050x1000 gives 62 rows by 64 columns.
  - Then it is patchified.
  - Anchor rows are 0.999 x clean + (1 - 0.999 in fp32) x the anchor noise.
- **The noise.** One CPU torch generator per request (`moss_image/noise.zig`, its draws chained), drawn in order: the
  target, each reference's anchor noise, then each reference's posterior noise on its full grid.

```sh
zig build tf-h3-image-encoder -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-encoder MODEL_DIR reference.png SEED out/ref [CANVAS]
zig-out/bin/tf-h3-image-encoder selftest
```

It writes:

- `out/ref.rows.f32`: the clean rows [rows, 96];
- `out/ref.moments.f32`: the blended moments [48, h, w], before sampling;
- `out/ref.grid.json`.

With `CANVAS` (WxH), the generator first draws the target's noise and this reference's anchor noise, as a
one-reference edit does, and `out/ref.anchors.f32` is written too. Without it, the posterior's noise is the seed's
first draw.

`selftest` runs each kernel on small synthetic weights against a float64 host reference and exits 1 on a miss. It
covers:

- a reflect-padded 3x3 convolution and the stride-2 downsample;
- a 1x1 convolution with a residual;
- GroupNorm with SiLU;
- both kinds of ResNet block;
- a small encode over uneven tiles, run twice.

Its gates, measured on an M3 Ultra:

| Check | Gate | Measured |
|---|---|---|
| Each layer, worst error over the largest value | 1e-5 | 2e-7 to 9e-7 |
| The tiled moments | 1e-4 | 1.6e-6 |
| The fp16 samples | one fp16 step at the largest value | within it |
| Repeat runs | bit-identical | bit-identical |

A 1024 x 1024 reference at the released widths encodes in 0.6 s.

## Vision tower

`tf-h3-image-vision` makes an edit reference's rows for the text encoder: Qwen3-VL's vision tower with DeepStack, on any
Apple GPU. Its kernels compile from source when it loads, and it uses no tensor units.

```sh
zig build tf-h3-image-vision -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig-out/bin/tf-h3-image-vision TEXT_ENCODER_DIR MODEL_DIR reference.png out/ref.vision.safetensors
zig-out/bin/tf-h3-image-vision selftest [WORK_DIR]
```

**What runs:**

1. The picture (PNG, JPEG or WebP) is read upright in RGB. Its Qwen copy has a short edge of exactly 1,024: the
   original when it already has one, else both sides scaled, snapped to 32 (halves to even) and resampled with Pillow's
   LANCZOS (`picture_prompt.zig`'s rule, in `qwen_copy.zig`).
2. The processor in `MODEL_DIR/processor/preprocessor_config.json` (Qwen2VLImageProcessorFast) cuts it into patches
   (`core/picture/patches.zig`): a smart resize to multiples of 32 inside its pixel bounds, a no-op on a snapped copy;
   bytes / 255, then mean and std 0.5; patches of 16 x 16 with the frame repeated, in merge order. A 1024 x 1024
   reference gives 64 x 64 patches and 1,024 tokens.
3. The tower is core's Qwen-VL tower (`core/qwen_vl`), the one Qwen3.5 serves with:
   - the patch projection, plus four bilinear taps of the learned 48 x 48 position grid;
   - 27 blocks of 1,152 with 16 heads of 72, a 2D axial rotary and a gelu_pytorch_tanh MLP of 4,304;
   - the merger: LayerNorm per patch, the 2 x 2 patches side by side, then fc1, exact GELU and fc2 to 5,120.
4. **DeepStack.** After blocks 8, 16 and 24 (`deepstack_visual_indexes`), each tap's own merger reads the residual.
   Its LayerNorm runs after the shuffle, over a merged row's 4 x 1,152 = 4,608 values; then fc1, exact GELU and fc2
   to 5,120, each with its bias.

It writes `grid` I32 [3] (1, rows and columns in patches), `rows` BF16 [tokens, 5,120] and `deepstack` BF16 [3,
tokens, 5,120], aligned. `tf-h3-image-prompt --reference` reads it.

**The checkpoint.** `TEXT_ENCODER_DIR` is MiniMax-H3's `text_encoder/`, an HF `Qwen3VLForConditionalGeneration`. The
tower's prefix is whatever precedes `blocks.0.norm1.weight` in its index, such as `model.visual.`. The loader reads
exactly the 351 bf16 tensors of a Qwen3-VL tower with three DeepStack mergers (`deepstack_merger_list.{0,1,2}` with
`norm`, `linear_fc1` and `linear_fc2`), and refuses a checkpoint with more or fewer. The patch kernel is read in
torch's [1,152, 3, 2, 16, 16] layout. These names and shapes come from the HF model's code and config; the release's
`text_encoder/` loads under them, as `model.visual.`, on an M5 Ultra.

**Precision.** As for Qwen3.5's images:

- bf16 weights and patches;
- products in simdgroup matrices with fp32 sums, each rounded once to bf16;
- the residual stream in fp32, where the reference's bf16 tower keeps bf16;
- LayerNorm with fp32 statistics and eps 1e-6, rounded once;
- q and k turned in fp32, then rounded;
- fp32 scores, softmax and sums.

**Qwen3.5 is unchanged.** DeepStack is an opt-in. `Config.parse` still refuses a tower with taps, and only
`Config.parseDeepStack` reads them. A tower without taps loads the same tensors under `vision_tower.`, into the same
buffer, and runs the same kernels in the same order.

**Selftest, gates fixed before the first run.** `selftest` writes a tiny Qwen3-VL tower in HF's names and layouts under
`model.visual.`, beside a text-model tensor the loader must leave alone:

- 4 blocks of 144, with 2 heads of 72 and an MLP of 272;
- a 64-position grid, a 2 x 2 merge to 200, and DeepStack taps after blocks 0, 2 and 3.

It parses its config and processor, finds its prefix and loads it as a run does. A 288 x 192 picture goes through the
processor's patches: 12 x 18 patches, 54 tokens. A float64 host reference runs the whole tower at the same bf16
rounding points: the patch projection, the learned grid's taps, the rotary, attention, the MLP, the merger and each
DeepStack merger. The gates:

- **Each output** (the merged rows and each tap's rows): within twice a float32-sum host run's distance from float64,
  in both 1 - cosine and relative error, with cosine at least 0.999.
- **A second run:** byte for byte the same.
- **No taps asked:** the same merged rows.
- **The Qwen3.5 path:** the same tower as MLX holds a Qwen3.5 one (`vision_tower.*`, the patch kernel [O, T, Y, X, C],
  no DeepStack), loaded by `Tower.load` and run by `image()`, gives the same merged rows.

On an M3 Ultra:

| Output | Cosine | Relative error | Float32-sum host run |
|---|---|---|---|
| The merged rows | 0.9999825 | 5.92e-3 | 0.9999828, 5.87e-3 |
| DeepStack after block 0 | 0.9999949 | 3.20e-3 | 0.9999950, 3.17e-3 |
| DeepStack after block 2 | 0.9999838 | 5.69e-3 | 0.9999832, 5.80e-3 |
| DeepStack after block 3 | 0.9999841 | 5.64e-3 | 0.9999837, 5.71e-3 |

Each output sits 0.96 to 1.02 times as far from float64 as the float32-sum run. The second run, the untapped run and
the Qwen3.5 path are byte-identical. The gates passed on the first run, unchanged.

`zig build test-h3-image-prompt` also runs the tower's host tests: both parses of the config, the 351-tensor plan
beside the 333 of Qwen3.5's, the prefix search, the scratch plan and the reference's norm and GELUs.
`zig build compile-h3-image-vision` compiles the tool without running a shader.

## Checkpoint

`MiniMaxAI/MiniMax-H3-Image`. `transformer/` is bf16, with the patch projection, time embedder and step heads in
float32. The text encoder is MiniMax-H3's Qwen3-VL; its 50th layer's rows are the prompt's. The weights are under the
MiniMax-H3-Image Community License, which excludes the European Union, the United Kingdom, the Republic of Korea and
the United States of America. Read it before use.

## Not done

- **Checked against the release:** text to image runs end to end on the release weights
  ([On an M5 Ultra](#on-an-m5-ultra-with-the-release-weights)); the reference cases and the gates above still wait for
  a reference pipeline's case.
- **Edits:** they run end to end in five tools, not one. With several references, each reference's noise is drawn
  as a one-reference edit draws it, not on the edit's one chain. The vision tower's selftest misses one bound on an
  M5. Its 1,024-token pass at the released size takes 0.25 s there.
- **Serving:** no `/v1/images/generations` or `/v1/images/edits` route for it, and no CUDA path.
- **Machines:** the transformer runs on an M5 only; chips without tensor units are refused.

## Host checks

`zig build test-h3-geometry` runs the picture rules without a GPU:

- the canvas, including 1280x720 drawn at 2720x1536 and 4000x3000 at 4000x3008;
- the schedule's float32 round trip and its refusals;
- positions, the rotary layout, the packing and the tile map;
- the model folder's checks;
- the case reader's picture admission.

`zig build compile-h3-host` compiles both H3 tools without the Metal compiler.

`zig build test-h3-image-prompt` also checks the read-ahead: spans across pieces and files are read once, and a short
or missing file fails.

## Kernel contract

The edit encoder's, prompt path's and image decoder's kernels are registered with the kernel contract
([kernel-contract.md](../kernel-contract.md)): 31 registrations with 34 planted controls, and every launch's limits.
They are checked against float64 at their rounding points, beside each tool's own `selftest`. `zig build
h3-image-contract` runs them on any Mac, with no Metal compiler; `tf-kernel-contract` runs them with every other kernel.
On an M3 Ultra all 31 pass gates 1 to 4 and every control goes red as planted.

The contract found one fault, now fixed: the prompt path's attention filled a row's last block of 4 keys with its last
key at weight 0, so an infinite value there gave 0 x Inf = NaN where float64 keeps the infinity. A padded key now adds
nothing; finite outputs keep their bytes.

## Speed: a byte-identical cold-start win

A first picture after the weights leave the page cache spent most of its time waiting on disk, not the GPU. Each tool
maps its bf16 weights and Metal reads them in place, so the first touch of every 16 KiB page is a fault served by one
small read. On the M5 Ultra's SSD a 2.3 GB file read that way takes 40 s (0.06 GB/s); 16 threads each `pread`ing 8 MiB
pieces take 0.7 s (3.2 GB/s). Faulting the mapping from 16 threads stays at 0.38 GB/s.

So each tool now reads the bytes it is about to use into the page cache first (`h3/read_ahead.zig`), then runs
unchanged on the same mapping: the transformer reads its whole shards when `Model.load` opens them; the prompt path
reads its 50 text layers, embedding, `condition_proj` and token refiner; the decoder reads its tensors before its
bf16 conversion. A missing or short file fails the load. The load stays zero-copy; only the faults are gone.

**Card 35 ships the cold-start improvement:** 229 to 61 s at 2048x2048 and 288 to 123 s at 4096x4096.
The lead's 12 November scope decision keeps the unchanged native path as the strict quality oracle and makes
cross-stack picture differences descriptive, not an acceptance gate. No further engine change is part of this round.
Card 50's resident server (above) brings warm 2048 to **16.4-16.6 s**, level with that diffusers/MPS path; see
[Resident, measured](#resident-measured). Card 51 holds its weights resident and brings it to **12.7-13.0 s**; see
[Weights held resident, measured](#weights-held-resident-measured).

### Gates, fixed before the runs

Same machine, build flags, seed 42, prompt (`prompt.txt` of card 34) and sizes for both builds. The case, final
latent and PNG of the new build must be **byte-identical** to the old build's: the change only decides when pages
are read, so any difference is a bug. Both sizes passed: `cmp` found no difference in any of the three files.

### Measured

Before each run, every weight file (`text_encoder/`, `transformer/`, `vae/`) was evicted from the page cache
(`msync(MS_INVALIDATE)` on a mapping). Wall seconds per tool, as `date` around each process; the last column is the
picture.

| Size | Build | Prompt | Transformer (load) | Decoder | Picture |
|---|---|---|---|---|---|
| 2048 | retained baseline, binary revision unrecorded | 87 | 116 (102.9) | 26 | 229 |
| 2048 | read ahead | 29 | 28 (12.7) | 4 | **61** |
| 4096 | retained baseline, binary revision unrecorded | 45 | 135 (75.6) | 108 | 288 |
| 4096 | read ahead | 13 | 78 (16.9) | 32 | **123** |

A cold picture is 3.8x faster at 2048 and 2.3x at 4096. The steps did not change (1.83-1.86 s at 2048, 9.9-10.1 s
at 4096), and are now most of the 4096 picture. The retained baseline prompt at 4096 found part of the encoder still cached from
the 2048 run before it (its encoder took 41 s against 76 s at 2048), so its gain is understated.

### Other stacks, measured here

The comparison stacks in the approved scope are diffusers on MPS and ComfyUI's int8 repack. The native cold-start
win above is measured against our unchanged path, not a claim to beat every stack on every workload. The earlier
cross-stack timings below are retained as diagnostics: diffusers included a custom checkpoint conversion, and
ComfyUI's warm 2048 used seed 43 rather than 42. The endpoint includes PNG saving:

| Stack | 2048 first picture | 2048 warm | 4096 warm | Seconds per step, 2048 / 4096 |
|---|---|---|---|---|
| TensorFold, read ahead (diagnostic; binary revision unrecorded) | 61 | 27 | 100 | 1.83 / 10.0 |
| diffusers with custom checkpoint conversion on MPS | 358.6 (load/conversion 333.2) | 20.0 | 167.7 | not timed |
| ComfyUI `b2f6707` (int8 DiT and text encoder, fp16 VAE) | 148 | 32.2 (seed 43; excluded from warm comparison) | 149 | 3.5 / 17.2 |

The documented-loader and matched-input reruns below supersede the converted diffusers runner and seed-43 warm
ComfyUI comparison. Native warm 4096 took 100 s versus documented diffusers' 103.640 s repeat; this small observed
advantage is a single-run comparison, not a noise-controlled speedup. No warm-2048 improvement is claimed here.

How each ran:

- **diffusers** ([docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)):
  `MiniMaxH3ImageModularPipeline`, diffusers `0905b667fe3f37f43b6ef9c6e05ad20636206938` (`0.42.0.dev0`),
  torch `2.15.0`, transformers `5.22.2`, safetensors `0.8.0`, Pillow `13.0.0`; components resident on MPS.
  The local runner loaded the text encoder, tokenizer, processor, VAE and scheduler, then built the transformer
  with `state_dict("mps")`: QKV splitting, concatenation, key renaming and gate/up rearrangement were inside the
  333.2 s load timer. That number cannot be attributed entirely to page faults and cannot represent the unchanged
  documented loader. The first request took 24.2 s plus 1.2 s to save PNG (358.6 s including load); the warm request
  took 19.2 s plus 0.8 s to save PNG. The 4096 request took 142.8 s plus 24.9 s to save PNG.
- **ComfyUI** ([tutorial](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image)): its text-to-image
  template with the expander off, posted to a headless server. On a Mac `comfy_kitchen` has only its eager backend:
  the int8 weights are dequantized before each product, so they halve the bytes read but not the work, and the text
  encoder runs on the CPU.
- **SGLang and vLLM-Omni** ([SGLang cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3-image),
  [vLLM-Omni recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3-Image)): CUDA and ROCm only, so neither runs on a
  Mac. SGLang's warm 2048 is 1.934 s on an H200 and 16.21 s on a DGX Spark (GB10), excluding startup and PNG
  encoding. It keeps components resident, precomputes the AdaLN tables (as `modulation.zig` does), fuses QK-norm with
  rotary, runs dense attention, batches decode tiles, caches text conditioning, and offers FP8, CUDA graphs and
  multi-GPU splits.

Identical seed numbers alone are insufficient for picture comparison: unpacking the native case's noise to
`[1, 24, H/16, W/16]` and comparing it with torch `2.15.0` CPU `randn` at seed 42 finds a maximum absolute difference
of `2.205371856689453e-6` at both sizes. The reruns below inject the exact native initial latent, preserve the prompt
and schedule, and retain numerical final-latent and decoded-RGB results. ComfyUI repeats keep seed 42 and disable
execution caching, rather than change the seed to force execution.

The branch's merge base is the card-34 fork at `49e08ffd4ab70802d291324a314a4cdc00efa25f`; the retained baseline
executables do not have an independently recorded source revision. The lead removed the unidentified maintainer
image stack from this card's comparison scope: upstream HEAD `eab305db0091a23b28122caadc8bf069f53d211e` has neither
this image recipe nor the image decoder module. No result against that stack is claimed.

#### Review rerun protocol

Before the cross-stack reruns, the diagnostic bounds were fixed at final normalized latent cosine **>= 0.99** and
decoded RGB maximum absolute difference **<= 1**, with finite latents and matching shapes. The image bound failed;
those failures remain recorded below. On 12 November, the lead explicitly replaced cross-stack pass/fail acceptance
with descriptive PSNR and SSIM, since different stacks have different rounding. This is a scope decision, not a
retroactive pass under the old bound. The native optimization gate remains byte identity for case, latent and PNG.

`tools/zig/h3_image_compare.py` compares raw normalized F32 latents, or ComfyUI `SaveLatent` files, and decoded PNGs:

```sh
python tools/zig/h3_image_compare.py baseline.latent.f32 candidate.latent.f32 baseline.png candidate.png --size 2048
python tools/zig/h3_image_compare.py baseline.latent.f32 candidate.latent baseline.png candidate.png --size 4096
```

It exits nonzero on a failed gate. Both retained native sizes give cosine 1, relative error 0, pixel maximum 0 and
pixel MAE 0; separate byte comparisons also pass. A planted negated latent and black image fail both gates
(cosine -1, pixel maximum 255). These checks validate the comparator, not a new cross-stack speed claim.

The normal diffusers loader resolves `transformer_diffusers/` and `vae_diffusers/` from the pinned checkpoint
`fbde899fffb5c5225ee8dd7027680113746af42e`. Its required replacement invocation is:

```python
pipe = ModularPipeline.from_pretrained(model_dir)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("mps")
torch.mps.synchronize()
```

No `state_dict` conversion belongs in this comparison. Time initialization through the final synchronization,
then each complete request through `image.save(...)` returning. Save latent receipts outside the request timer.
The pinned files finished downloading and passed the fetcher's SHA-256 verification before the normal-loader run
below. The OS page cache was not evicted after download and verification; this is process-cold, not storage-cold.

#### Documented diffusers loader, matched inputs

Same versions and machine as above, Python 3.12.16. All components loaded with the documented calls above,
without checkpoint conversion, and remained resident on MPS. The command was
`nice -n 10 "$DIFFUSERS_VENV/bin/python" diffusers_documented.py`; the local measurement runner used this loop
after loading the pipeline (paths supplied locally; no weights or output images are published):

```python
for size in (2048, 2048, 4096, 4096):
    noise = load_file(str(cases / f"base-{size}-a.safetensors"))["noise"]
    noise = noise.reshape(size // 32, size // 32, 24, 2, 2)
    noise = noise.permute(2, 0, 3, 1, 4).reshape(1, 24, size // 16, size // 16).contiguous()
    start = time.perf_counter()
    state = pipe(prompt=prompt, height=size, width=size, latents=noise,
                 generator=torch.Generator().manual_seed(42), output=["images", "latents"])
    torch.mps.synchronize()
    image_done = time.perf_counter()
    state.get("images")[0].save(output_png)
    png_done = time.perf_counter()
    # Outside the request timer; this output is already unpacked NCHW.
    state.get("latents").float().cpu().numpy().tofile(output_latent)
```

Initialization from `from_pretrained` through MPS synchronization took **129.383 s**; startup including Python
imports from the runner's first timestamp took **144.460 s**. First timestamp through first PNG written was
**169.447 s**, including preparation of the matched noise. These are not the old 333.2 s conversion measurement.

| Size / request | Inference through MPS synchronization | PNG save | Request through PNG written |
|---|---|---|---|
| 2048 first | 24.122 s | 0.695 s | 24.817 s |
| 2048 repeat | 15.806 s | 0.827 s | 16.633 s |
| 4096 first, after 2048 | 91.225 s | 10.813 s | 102.038 s |
| 4096 repeat | 97.944 s | 5.696 s | 103.640 s |

All requests used seed 42 and the exact native initial noise; each executed six steps. Both repeats produced
identical final latent values and decoded pixels. The initial capture runner mistakenly unpacked the already-NCHW
output a second time; that pure permutation was inverted before comparison, without changing any values. The
snippet above shows the correct direct capture. Against the native baseline, both repeats at each size give:

| Size | Latent cosine | Latent relative error | RGB maximum absolute difference | RGB MAE | Fixed diagnostic gate |
|---|---|---|---|---|---|
| 2048 | 0.9965921509572248 | 0.08261069531715759 | 237 | 3.2704102198282876 | **FAIL image**, pass latent |
| 4096 | 0.9960658854188926 | 0.08869195143972416 | 255 | 3.564861317475637 | **FAIL image**, pass latent |

This closes the missing normal-loader measurement. It does not make the old cross-stack pixel diagnostic pass.
At 2048, native storage-cold 61 s is below diffusers process-cold 169.447 s despite the latter's un-evicted page cache.
There is no standalone process-cold 4096 diffusers run: adding its measured 129.383 s load to the 102.038 s first
4096 request gives **231.421 s as an estimate**, not a measured first-picture total. The native 123 s storage-cold
4096 result must not be presented as a directly measured cold-versus-cold win over that estimate.

#### ComfyUI matched-input diagnostic rerun

ComfyUI `b2f670744386d36d39c41cc37cd1e30151046222`, Python `3.12.16`, torch `2.15.0`, transformers `5.22.2`,
safetensors `0.8.0`, Pillow `13.0.0`, comfy-kitchen `0.2.41`; macOS `27.0` build `26A425`, M5 Ultra, 256 GiB.
The template used the same int8 transformer/text encoder and fp16 VAE, expander off, Euler/simple, six steps, CFG 1.
The server ran with `--cache-none`; every history entry recorded `execution_cached: {nodes: []}` and every run
executed six steps. All four saved PNG prompt metadata records contain seed **42**.

Only the benchmark noise source was replaced, before starting ComfyUI's `main.py`; no sampler or model arithmetic
was changed. For each native case, the hook returned this exact tensor from `comfy.sample.prepare_noise`:

```python
noise = load_file(f"base-{size}-a.safetensors")["noise"]
noise = noise.reshape(size // 32, size // 32, 24, 2, 2)
noise = noise.permute(2, 0, 3, 1, 4).reshape(latent.shape).contiguous()
assert seed == 42 and noise_inds is None and noise.dtype == latent.dtype
return noise
```

`SaveLatent` captured sampler output and `SaveImage` captured decoder output. ComfyUI's H3 latent format has scale
factor 1; comparison uses its normalized `[1, 24, 1, H/16, W/16]` tensor directly against the native F32 latent.
The two repeats at each size produced identical latent values and decoded pixels. PNG file hashes differ because
the recorded output-prefix metadata differs.

| Size | Latent cosine | Latent relative error | RGB maximum absolute difference | RGB MAE | Fixed diagnostic gate |
|---|---|---|---|---|---|
| 2048, both repeats | 0.9947039242567004 | 0.10285523962274952 | 255 | 4.653071403503418 | **FAIL image**, pass latent |
| 4096, both repeats | 0.9931320414034048 | 0.11718428606255665 | 255 | 5.589069624741872 | **FAIL image**, pass latent |

The observed client intervals were 123.944 / 46.855 s at 2048 and 149.556 / 186.907 s at 4096. These include HTTP
polling and latent saving as well as PNG saving, and `--cache-none` reloads graph components; they are **not** the
resident-warm or process-cold timing boundaries above. Downloads shared the disk during these diagnostics. Do not
use these intervals for a speed ranking. The old pixel diagnostic remains failed, not relabelled as passed. The
earlier ComfyUI first-picture 2048 observation was 148 s versus native 61 s; no independent process-cold 4096
ComfyUI total was recorded. These limitations do not change the measured native before/after cold-start win.

#### Cross-stack picture differences (descriptive, not gates)

Measured on 12 November from the retained matched-noise PNGs above, not from a new inference run. Each pair uses
seed 42, card 34's prompt, the same native initial noise and six steps. Native baseline and read-ahead case, latent
and PNG bytes were rechecked equal at both sizes. Each stack's two reruns also decode to identical RGB pixels, so
both repeats give the same metrics. These values neither imply pixel equality nor replace the native byte gate.

| Native picture versus | Size | PSNR (dB) | SSIM |
|---|---|---|---|
| Documented diffusers/MPS | 2048 | 30.150806 | 0.979374 |
| Documented diffusers/MPS | 4096 | 28.598143 | 0.972045 |
| ComfyUI int8 repack | 2048 | 28.009878 | 0.966418 |
| ComfyUI int8 repack | 4096 | 26.307662 | 0.959728 |

Decode PNGs with Pillow `13.0.0` to RGB uint8 without resizing, linearizing or color-profile conversion. PSNR is
`10 * log10(255**2 / MSE)`, pooling all RGB samples. SSIM averages the three channel scores, using an 11x11 Gaussian
window, sigma 1.5, population covariance, K1=0.01 and K2=0.03, with no downsampling; the five-pixel border is excluded
from the mean by scikit-image. Measurement versions: Python `3.12.16`, NumPy `2.5.4`, SciPy `1.18.1`, scikit-image
`0.26.0`. Reproduce each row with the corresponding pair of retained PNGs:

```python
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

a = np.asarray(Image.open(native_png).convert("RGB"))
b = np.asarray(Image.open(stack_png).convert("RGB"))
assert a.shape == b.shape
print(peak_signal_noise_ratio(a, b, data_range=255))
print(structural_similarity(a, b, channel_axis=-1, data_range=255, win_size=11,
                           gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
```

Local receipt names: native `base-{size}-a.png` (equal to `new-{size}-a.png`); documented diffusers
`documented-{size}-{1,2}.png`; ComfyUI `matched-{size}-{1,2}_00002_.png`. No weights, latents or pictures are published.

### Resident, measured

Card 50, on the M5 Ultra above, ReleaseFast, card 34's `prompt.txt`, weights already in the page cache. One server
process served three 2048x2048 requests: seed 42, seed 42 again, then seed 7. Seconds from its timing lines:

| Request | Prompt | Case | Transformer | 6 steps | Decoder (PNG) | Picture |
|---|---|---|---|---|---|---|
| startup, once | | | | | | 29.7 (all three loads) |
| 1, seed 42 | 3.42 | 1.07 | 2.15 (bound) | 11.14 | 1.81 | 19.59 |
| 2, seed 42 | 1.97 | 0.01 | 0.00 (reused) | 12.61 | 2.04 | **16.63** |
| 3, seed 7 | 1.18 | 0.16 | 0.00 (reused) | 12.37 | 2.67 | **16.38** |

No request after startup loads a model: the prompt's 50 text layers on the GPU take 0.30 s of its 1.1-3.4 s, the
rest is tokenizing and host work. The one-shot tools spend 9.4 s, 17.3 s and 2.4 s loading on every picture.

**Gate: byte-identical to the one-shot tools.** Requests 1 and 2 against the one-shot `tf-h3-image-prompt`,
`tf-h3-image ... free` and `tf-h3-image-decoder` at seed 42: `cmp` found no difference in the prompt case, the
transformer case, the latent (sha1 `f3563c56…`) or the PNG. Request 3 (reused binding, new seed) matches the one-shot
latent at seed 7 (sha1 `04e3c4f1…`).

### Weights held resident, measured

Card 51, same machine, ReleaseFast, card 34's `prompt.txt`, seed 42, 2048x2048, weights in the page cache. Each build
ran one server process with five requests back to back; the first binds the transformer, the other four reuse it.
The unchanged build was rebuilt from `origin/main` and run just after the new one, at a load average of 5.6 to 8.8:

| Build | Request | Prompt | Step 1 wall (GPU) | 6 steps | Decoder (PNG) | Picture |
|---|---|---|---|---|---|---|
| unchanged | 1, bound | 0.39 | 1.86 (1.84) | 11.09 | 1.40 | 13.67 |
| unchanged | 2-5, reused | 0.37-0.38 | 2.20-2.24 (1.85-1.87) | 11.45-11.49 | 1.39-1.40 | 13.24-13.28 |
| weights resident | 1, bound | 0.28 | 1.87 (1.85) | 11.12 | 1.42 | 13.02 |
| weights resident | 2-5, reused | 0.22 | 1.85-1.86 (1.85) | 11.10-11.15 | 1.38-1.42 | **12.70-12.77** |

Warm runs 2-5 of the resident build: 12.77, 12.75, 12.72 and 12.70 s, all under the documented diffusers/MPS 2048
repeat's 16.633 s. Step 1's wall now equals its GPU time, and the prompt's 50 layers take 0.21 s of wall for 0.21 s
of GPU. Earlier the same day, at a load average of 28 to 40, the unchanged build's warm requests took 13.75-15.45 s
(prompt 0.51-2.08 s, step 1 2.55-2.63 s wall); card 50's 16.38-16.63 s above was another such run. The set removes
most of that spread: the resident build's warm requests were 12.71-13.11 s across two runs.

**Gate: byte-identical to the unchanged build.** All four warm latents and PNGs of each build are `cmp`-equal, and
equal to card 50's one-shot seed-42 latent. `tools/zig/h3_image_compare.py` gives latent cosine 1.0, relative error
0, pixel maximum 0. `zig build test-h3-image-server` runs the resident side of its parity test on queues holding the
checkpoint's residency set.

### Next, ranked

| # | Idea | Size | Basis |
|---|---|---|---|
| 1 | SGLang's conditioning cache | under -2 s at warm 2048 | Residency landed with card 50: warm 2048 went from 27 s to 16.4-16.6 s, more than the 6.3 s of loader logs the projection counted, since each one-shot tool also re-reads its files and rebuilds its kernels. A conditioning cache would skip only the prompt (1.2-2.0 s warm); it needs separate accounting and cannot omit prompt work in an uncached comparison. |
| 2 | PNG writing | upper estimate -18 s at 4096, -4 s at 2048 | Time after decoder tiles includes single-threaded deflate (`moss_image/png.zig`). Parallel bands could remove much of it but change compressed bytes, so they do not satisfy this card's fixed PNG byte-identity gate. Any separate decoded-pixel gate requires explicit approval before runs. |
| 3 | int8 weights and activations on the M5's tensor units | about -30 s of the 60 s of steps at 4096 | The steps are compute-bound: a 2048 step is about 183 TFLOP in 1.83 s, a 4096 step about 1,020 TFLOP in 10 s, both near 100 TFLOPS of bf16. int8 `matmul2d` at twice bf16's rate would halve the GEMMs. ComfyUI's int8 here shows the other half: dequantized before each product, its steps are 1.7-1.9x slower than our bf16, so int8 weights alone buy only bytes. Not bit-identical; it needs its own latent gate. SGLang's FP8 on an RTX PRO 6000: -32% at 2048, -22% at 4K. |
| 4 | Attention | up to -15 s at 4096 | At 16,408 rows attention is about 38% of a step's work, growing with the square of the rows: 4096's step is 5.4x 2048's for 4x the rows. SGLang's fused QK-norm and rotary saves under 1% of a step here. |
| 5 | Step caching (TeaCache, FBCache, cache-dit) | -10 s per skipped step at 4096 | vLLM-Omni quotes 1.5-2.0x for many-step models and less for few-step ones; SGLang's recipe runs all six steps. Each skip is a sixth of the steps; it needs a gate against the six-step latent. |
| 6 | Decoder tiling | under -2 s | Its 441 tiles already take 4.1 s of GPU; SGLang batches its tiles the same way. |

## On an M5 Ultra with the release weights

Measured on TensorFold 1.2.7 at `a1c758603` (`v1.2.7-154-ga1c758603`, this branch with our main and card 43 merged in),
built with `-Dcpu=apple_m1 -Doptimize=ReleaseFast`. Text to image from the release folder (`text_encoder/`,
`transformer/`, `vae/`; not the diffusers copies), seed 42, a 24-token prompt, each tool its own process, no other
model running; other checkouts' builds shared the CPU and disk. Two runs at each size, 2048 then 4096, back to back;
each tool maps its weights again. Both runs at a size write the same bytes as the first build's PNGs, kept outside the
repo as `h3-image-runs/34/fox-2048-seed42.png` and `fox-4096-seed42.png` beside it.

Seconds; a step's wall time, then its GPU time in brackets.

| Stage | 2048 run 1 | 2048 run 2 | 4096 run 1 | 4096 run 2 |
|---|---|---|---|---|
| Prompt path: map, 50 layers, refiner | 27.0 (0.53) | 35.5 (0.49) | 10.7 (0.47) | 14.0 (0.52) |
| Transformer load | 21.9 | 19.4 | 21.9 | 14.7 |
| Step 1 | 1.88 (1.83) | 1.88 (1.83) | 9.95 (9.81) | 10.01 (9.82) |
| Step 2 | 1.84 (1.84) | 1.83 (1.83) | 9.86 (9.85) | 9.85 (9.83) |
| Step 3 | 1.84 (1.83) | 1.83 (1.83) | 9.84 (9.83) | 9.86 (9.85) |
| Step 4 | 1.84 (1.84) | 1.83 (1.83) | 9.84 (9.83) | 9.87 (9.86) |
| Step 5 | 1.83 (1.83) | 1.83 (1.83) | 9.81 (9.80) | 9.83 (9.81) |
| Step 6 | 1.83 (1.83) | 1.83 (1.83) | 10.05 (9.79) | 9.95 (9.80) |
| Decoder: weights, 121 / 441 tiles, PNG | 10.9 (tiles 1.17) | 20.8 (tiles 1.18) | 96.9 (tiles 4.19) | 102.6 (tiles 4.25) |
| Total a picture | 71.4 | 88.0 | 189.4 | 191.0 |

Peak memory footprint per tool: prompt 0.33-0.41 GB, transformer 1.41-1.42 GB at 2048 and 4.40 GB at 4096, decoder
1.31 GB and 1.51-1.52 GB. The weights are read in place, so the machine's used memory rises by about 48 GB for the
transformer and 33 GB for the text encoder while each runs. Wall time outside the steps is mostly weights read from
disk (the decoder's 6.9-15.1 s at 2048, 26.4-48.8 s at 4096), and at 4096 the decoder spends 36-66 s past its tiles
writing a 38 MB PNG.

The maintainer's gates that run here:

| Gate | Result |
|---|---|
| `tf-h3-image-prompt selftest` | pass: kernels 99.97-100% bit-equal; whole runs 0.77-1.03 x the float32-sum host's relative error, 0.57-1.06 x its 1 - cosine |
| `tf-h3-image-encoder selftest` | pass: layers 2.0e-7 to 8.6e-7 (bound 1e-5); tiled moments 1.6e-6 (bound 1e-4) |
| `tf-h3-image-decoder selftest` | pass: 24 blocks relative 1.49e-2, worst 16.5 ULP; picture bytes at most 1 apart |
| `tf-h3-image-contract` | pass: 31 of 31 kernels, 34 of 34 controls red, no limit problems |
| `tf-h3-image-vision selftest` | **fail**: DeepStack after block 0, 1 - cosine 1.32e-5 against a bound of 1.00e-5 (twice the float32-sum host's 5.0e-6); relative error 5.14e-3 passes (bound 6.34e-3) |
| `tf-h3-image` gates on a reference case | not run: no reference pipeline case exists yet |

The vision miss is the M5's: the M3 Ultra above passed the same output at 0.9999949, and on the M5 every tower output
sits further from float64 (merged rows 0.9999789 against 0.9999825). Only the M5 takes `nax.h`'s tensor-unit path, whose
`matmul2d` runs with `relaxed_precision`; the tower's fp32 score and value products go through it. Turning that flag
off is no fix: the cooperative tensors' element layout changes with it and every output breaks (cosine 0.75). The
tower's attention products need their own exact-fp32 path on the M5 before this gate can pass.
