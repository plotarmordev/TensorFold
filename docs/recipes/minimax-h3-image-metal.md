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
Each request prints one line of stage seconds: prompt, case, transformer (`reused` or `bound`), the steps' count
and seconds, decoder, total.
A reference-free prompt's states — its rows' ids, tags and the refiner's text rows — are kept across requests
keyed by the prompt's exact bytes, least recently used inside a byte budget (`H3_PROMPT_CACHE` MiB, 256 by
default; 0 keeps nothing). A repeated prompt skips the text encoder and refiner; an edit case's never leave the
cache alone, so `--reference` prompts always encode. The key is the prompt bytes alone because the tokenizer,
marks and geometry are checked at load and H3-Image has no negative prompt or CFG.

`--draft-steps N` runs each request's first N steps on the half-size canvas, mmh3's fast mode: the clean draft is
enlarged bilinearly, noised back to the next step's sigma (`--renoise-sigma X` overrides it) and redrawn at full
size for the remaining steps. A different picture than the six-step run, not a faster copy of it. A canvas side
must be a multiple of 64 pixels for the latent grid to halve; anything else refuses with H3BadCascadeGrid.
A bad request prints its error and the server reads the next line.
Start it with `profile` after `MODEL_DIR` to print each request's six steps by kernel: each launch runs in its own
encoder, timed by the GPU's timestamp counters at its start and end (`rows in`, `norm`, `norm quant`, `quantize`,
`q k v gate`, `heads`, `attention`, `attention out`, `mlp in`, `swiglu`, `mlp out`, `gate add`, `head`, `euler`). The
waits make the request slower than an unprofiled one, so the kernel seconds are for diagnosis, not for a speed gate.

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

**Precision.** Weights are bf16, read in place, but for fc1's int8 copy (see "int8 fc1, landed"). Products run in
simdgroup matrices with fp32 sums. Norms keep fp32 statistics, and the softmax and its sums stay in fp32. Each step
rounds to bf16 where the reference's bf16 arithmetic
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
zig-out/bin/tf-h3-image MODEL_DIR picture.safetensors out/case [once | free] [--draft-steps N] [--renoise-sigma X]
```

`MODEL_DIR` is the MiniMax-H3-Image folder: `model_index.json` (the image pipeline's class and partition),
`transformer/` (its config must say `audio_enabled: false`), `pdd_config.json` and the heads file. The tool:

1. checks the case's positions, rotary table, timesteps and Euler steps against ours;
2. runs each step on the reference's own input rows (teacher-forced) and compares the target's velocity;
3. runs all six steps from the noise with our own rows.

It writes `out/case.latent.f32`, the normalised latent [24, rows, columns] in float32, ready for the reference's
decoder. `once` runs the first teacher-forced step alone; `free` runs only the free run. It needs the M5's tensor
units, as the clip tool does.

`--draft-steps N` runs the first N steps on the half-size canvas, mmh3's fast mode: the clean draft is enlarged
bilinearly, noised back to the next step's sigma (`--renoise-sigma X` overrides it; off-grid steps blend the
distilled heads their sigma span covers, as ComfyUI's step_heads does) and redrawn at full size for the rest.
It draws a different picture, not a faster copy of the six-step one, so the teacher-forced and latent gates don't
apply: drafts only run the free pass (`once` with `--draft-steps` is an error), and cases with reference rows
refuse. It also writes `out/case.draft.h3case.safetensors`, the half-size draft's case.

Warm on this Mac (M5 Ultra, 2048x2048, the seed-42 fox prompt): six timing-valid steps take 10.4 s; the same six
steps with `--draft-steps 3` take 6.3 s — three half-size steps at 0.5 s each plus three full-size ones. The draft
draws a different but clean fox: latent cosine against the six-step run is 0.70 (a second six-step run's is 0.995),
matching the Spark port's 0.706. Side by side:

![six steps against draft-steps 3 on the same case](minimax-h3-image-metal/draft-vs-six-step.png)

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
Speech waits for its shared hold only behind a render's exclusive probe, which lasts microseconds and never spans GPU
work, so it never synthesizes unprotected. The prompt encoder (`tf-h3-image-prompt`, the server's first stage) waits
before each pass with three queued, and the decoder (`tf-h3-image-decoder`, its last) before each batch of tiles; both
print the seconds they waited.

### Speech contention receipt, 12 November 2026

Apple M5 Ultra, 256 GiB RAM, macOS 27.0 (26A425), Zig 0.17.0, ReleaseFast with `-Dcpu=apple_m1`.
The branch's isolated Drama 3 NVFP4 server served the Capri catalog in `synth.json`, CFG 1.5 with paired submission,
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

### Five voice notes during a resident render, 13 November 2026

Same machine, speech server and client, built from `b82e132` and served on port 18899; `tf-h3-image-server` drew
card 34's prompt at 4096x4096, seed 42. The server's first request after start, with its reference rows still unbuilt
in memory, reached first PCM in 2.879 s idle; the next took 0.414 s. Five notes then went out 15 s apart during the
render, all HTTP 200 with the render running at the start and end of each:

| Note | First PCM (s) | PCM bytes |
|---:|---:|---:|
| 1 | 1.375 | 245760 |
| 2 | 1.142 | 192512 |
| 3 | 0.687 | 225280 |
| 4 | 1.165 | 188416 |
| 5 | 1.355 | 225280 |

Steps 1-3, 5 and 6 waited 2.27-3.21 s each and reported `timing_valid=false`; step 4 waited zero and reported
`true`, with 10.04 s on the GPU like the others. The render resumed after each wait and exited 0; its latent matched
card 34's `t2i-4096.latent.f32` and its rows the 12 November baseline (`b3726693...`) byte-for-byte. The prompt
encoder and decoder are too short at this spacing to catch a note; `test-h3-image-server` has speech arrive during
each stage. The isolated server was stopped afterward; the live release server was not changed.

### Large resident renders: live signal check, card 64

The 14 November incident reported 17 consecutive 5472x3072 renders at about 85 seconds each and a voice-note
deadline failure. An 85-second render is not one uninterruptible denoising pass: the resident server already submits
individual transformer blocks, and the measured six steps below each took 11–12 seconds.

A live speech deployment must publish the same lock as the renderer. Updating only `tf-h3-image-server` cannot make
an uncoordinated speech process announce a request. Card 49's isolated-server receipts did not update the live
release server. Before diagnosing GPU submission granularity, observe both the lock and its generation during an
actual request to the endpoint used by the voice menu:

- Attempt a nonblocking exclusive `flock` on `/tmp/tensorfold-speech.lock` from a separate process every 20 ms;
  contention means speech holds it. Merely listing another user's open descriptors is not this check.
- Compare `/tmp/tensorfold-speech.lock.stamp` before and after the request. The coordinated server writes it even
  when acquiring the speech hold fails. Check cross-user access too: lock mode 0644, stamp mode 0666, length 16.
- Measure first PCM bytes **and the complete audio response**, not just HTTP headers. A streamed HTTP 200 can
  still miss a caller's deadline while delivering its body.

On the M5 Ultra, 256 GiB, with Zig 0.17.0 and `ea6a16d2e35783aacf6a69af40758785b67171ef`, a two-request
resident replay at 5472x3072, seed 2201, 183 prompt tokens completed in **76.08 and 85.58 seconds**. Both PNGs
were written and the process exited zero. The live Drama endpoint returned HTTP 200 and first PCM at **18.856 s**,
but completed its 618,496-byte response at **72.854 s**. The render was running throughout the voice request.
Across 776 exclusive-lock probes, none found a speech hold; the stamp remained unchanged. The first render's
six steps all reported zero wait and `timing_valid=true` despite overlapping speech. An idle voice probe likewise
observed no hold or stamp change, with first PCM at 16.438 s and complete audio at 20.805 s.

This establishes an absent live coordination signal, not the exact installed binary's revision. The endpoint
advertised version 1.2.6; its executable could not be inspected with the test account's permissions. The shared
files had the modes and length above. An old executable or a process-specific access/namespace restriction must
be resolved by the deployment owner. `timing_valid=true` cannot detect a nonparticipating speech process.
The required host/Metal test suite ran concurrently, so these are contention diagnostics, not clean throughput
benchmarks or a replay of all 17 original requests. The live service was not restarted or changed.

For a causal control only, the diagnostic client held a shared `flock` on the same file from immediately before
sending the voice request until the complete body was read. The unchanged image executable then paused step 1
for **6.84 seconds** and marked it invalid for timing. First PCM arrived in **1.804 s**, and the complete
618,496-byte response in **7.473 s**, with the renderer still running. Both queued images completed (85.85 and
89.54 s); `cmp` found both final latents and both PNGs byte-identical to the uncoordinated run. The executable
was rebuilt from the commit above with `zig build tf-h3-image-server -Dcpu=apple_m1 -Doptimize=ReleaseFast`;
its SHA-256 matched the measured executable. No arithmetic or scheduling change was needed for this control.

Both requests used `model="fishaudio/drama-3"`, `stream=true`, `response_format="pcm"`, and this public input:
“The little boat crossed the harbor before the sun came up, and the gulls were quiet over the water while the
fishermen coiled their nets along the pier.” The client waited for the renderer's `drawn on` line before sending.
The control's client-owned hold is **not** a production fix and does not demonstrate TOP-menu acceptance.
That deployment prerequisite was subsequently repaired; the follow-up below measures the live publisher without
a client-owned speech hold.

### Repaired live publisher and TOP voice menu, 15 November 2026

The render executable rebuilt from card 64's `206b911b5` has SHA-256
`4f18020216e7be43b2aeb90e651d86875dfd1d4147fddc79bb2857c5df6eb3ad`, identical to the diagnostic executable above.
No renderer arithmetic or scheduling change is needed: the missing component was the live speech publisher.
An idle request with the public boat text above now returned HTTP 200, first PCM in **1.105 s**, and the complete
618,496-byte body in **5.527 s**. Its generation changed and 46 of 47 nonblocking exclusive probes found a hold.

The actual TOP-menu client sends one JSON object `{"text":"..."}` on stdin to `ssh top-menu voice`, and receives
one base64-encoded M4A on stdout. This is a separate acceptance path from directly posting streamed PCM to the
Mac's speech endpoint. Measure from before SSH starts until its complete output is received and decoded;
require exit zero, valid audio and elapsed time below 20 seconds. Do not count early HTTP headers as success.
The observer uses only immediately released, nonblocking exclusive probes; it never holds a shared speech lock.

Five TOP-menu notes were sent after denoising step 2 of consecutive requests in the original 17-request replay.
Each uses the public boat text above, with a 20-second outer subprocess timeout. Each started and ended while
the same render was unfinished; every response decoded successfully with FFmpeg to mono 44.1 kHz AAC audio
in an M4A container (55,620 bytes, 6.635 seconds of audio). No test suite ran during this replay.

| Render in replay | Complete TOP reply | Contended / total lock probes | Generation changed |
| --- | ---: | ---: | --- |
| 7 | 9.397 s | 76 / 93 | yes |
| 8 | 9.538 s | 86 / 99 | yes |
| 9 | 9.774 s | 83 / 97 | yes |
| 10 | 9.769 s | 85 / 97 | yes |
| 11 | 9.676 s | 93 / 111 | yes |

The same TOP request can be reproduced without any client-owned lock:

```sh
printf '%s\n' '{"text":"The little boat crossed the harbor before the sun came up, and the gulls were quiet over the water while the fishermen coiled their nets along the pier."}' |
  ssh -o BatchMode=yes -o ConnectTimeout=10 top-menu voice
```

Decode the complete stdout from base64 and validate the audio. The SSH connect timeout is not a synthesis
deadline: the measurement harness separately enforces 20 seconds for the entire subprocess. Lock observations
and generation changes prove live synthesis occurred; the five deterministic audio files also have equal
SHA-256 `3eccc55d2ed7fcae1e8772aa6d7f4c91a404779de8dc2c6accde3d63c3ce303d`.

The original marketing 122 `r1/requests.txt` manifest contains **17 requests**, not 17 identical landscapes:
eight 5472x3072, six 3072x5472 and three 4096x4096 images. The unchanged prompts, seeds and order were replayed
through one resident process, with only output prefixes redirected into this worktree's `.zig-cache/card64/replay`.
The six prompt groups were camel (2201, 3201), tractor (1302, 2302, 3302), dog (1204, 2204, 3204), pool
(1308, 2308, 3308), ice-cream seller (1406, 2406, 3406) and orange (1205, 2205, 3205).

All **17 renders completed**, process exit zero, in **1,592.514 s** including startup. Individual renders took
74.45–111.49 s (median 92.33 s), including speech pauses. Every final latent and PNG matched its corresponding
original output by SHA-256: **17/17 latent and 17/17 PNG matches**, including both canvas orientations and squares.
The original camel/2201 latent hash was `70f97571c48288a58b61d99e2faa3cb6cd6946161e3437fc9bbdf3d84f5adf64`,
and its PNG hash `99582ded969f0c78866ceeb64ad10d9d72b5b1ec74e3f97d53025c32e40a23c9`.
The M5 Ultra's `ioreg` device and renderer utilization both sampled at 100% during the replay; this is a sample,
not continuous utilization telemetry. The platform was macOS 27.0, 256 GiB, Zig 0.17.0, ReleaseFast/apple_m1.

An additional direct streamed-PCM request was sent at the start of each render: **17/17 HTTP 200**, first PCM
1.061–2.224 s, complete body 5.484–9.421 s, each with an observed speech hold and changed generation.
The image logs recorded 183.28 s of denoising waits and excluded 25 of 102 steps with `timing_valid=false`.
Those waits include the five timed TOP notes above and two exploratory TOP notes during request 6; they are not
a speech-free throughput comparison. The renderer resumed without restarting or discarding an image.

The local receipts are `.zig-cache/card64/replay/{render.log,voice.jsonl,result.json}` and
`.zig-cache/card64/top-results.json`; the original manifest remains with marketing 122's run inputs.
Before the workload, `nice -n 10 zig build test test-golden` exited zero in 89.66 s (322/335 golden cases equal,
13 known differences, zero unexpected); `nice -n 10 zig build test-h3-image-server -Doptimize=ReleaseSafe`
exited zero in 3.27 s. That regression retains card 49's mid-encoder, refiner, decoder and denoising waits and
byte-identical output checks. The measured ReleaseFast executable rebuild also exited zero.

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
of 1,024, 16 heads of 64, 4 register tokens) on any Apple GPU. Its kernels compile from source when it loads, in one of
two builds of `image_decoder.metal`. With tensor units (the M5), `H3D_TENSOR` is set and `nax.h` inlined: card 152's
products on the tensor units, SwiGLU folded into w1's, vector attention loads. Every other Apple GPU compiles the
source as written, which is main's decoder before card 152 token for token (simdgroup-matrix products staged through
threadgroup memory, a separate `h3d_swiglu`), with main's dispatches. Card 152's products stay off pre-M5 GPUs:
`nax.h`'s simdgroup-matrix form could not be run on one, and forcing it on an M5 is no stand-in, since there
`nax.h`'s own `tf_frag_check` (`core/frags.zig`), compiled with `TF_SIMD_FRAGS`, gets 512 of 512 transposed sums and
510 of 512 plain ones wrong (21 Nov), while the tensor-unit form gets all of them right. The M3 Ultra receipts below
ran main's decoder, which is still the pre-M5 build.

**What runs:**

1. The latent file (normalised, [24, rows, columns] float32) is mapped back with `latents_mean` and `latents_std` in
   float32.
2. It is cut into tiles of 16 x 16 cells (256 pixels) overlapping by at least 64 pixels, as `picture_vae.zig` splits
   it.
3. Each tile runs `post_quant_conv`, then its cells become tokens through `x_embedder`, with the 4 registers and one
   zero token appended.
4. The 2D rotary table is built, and the 24 blocks run, then `norm_out` and `proj_out`.
5. Each token's 768 values are unpatchified to its 16 x 16 pixels. Up to 128 tiles run in one command buffer: a
   2048 x 2048 canvas's 121 tiles decode in one pass, the way mmh3 decodes its whole canvas at once.
6. Tiles are blended in float32: the top band against the tile above, then the left band against the tile to the left,
   both as decoded.
7. The result is mapped to [0, 1] with ImageNet's statistics, clamped and truncated to bytes.
8. If the request's size differs from the canvas, the picture is resized with Pillow's LANCZOS, then written as a PNG.
   At the canvas's size each tile row is its own pass, and once it is blended its final rows go, on a CPU thread, to
   the PNG's one zlib stream (`png.Stream`) while the GPU decodes the next tile row (card 170).

**Precision:**

- Every parameter is rounded from the file's F32 to bf16 when loaded, and the latent is rounded to bf16 at
  `post_quant_conv`.
- Products are bf16 (exact in the fragments' float32) with float32 sums, the bias added before the one rounding.
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

The transformer's products are registered too (`tf-kernel-contract --only 'h3 product'`): the 32x64-warp bf16 GEMM
in its whole and ragged forms bit-equal to MLX's gemm_nax that it replaced at every served shape and edge, fc1's int8
form against float64 at its scales at a 2048 picture's 4,120 and 4,096 rows, and `h3_quantize_rows` against its formula
at its 4,120 activation and 28,672 weight rows; each with its launch limits. On the
M5 Ultra all four pass gates 1 to 4. The contract found that the quantizer dropped NaN, since `max` ignores it: a NaN
activation took a finite scale and became an ordinary value. A group holding NaN now takes a NaN scale; finite inputs
keep their bytes.

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

### Products in 64-row-tile bands, measured

Card 81, same machine, ReleaseFast, card 34's `prompt.txt`, seed 42, one resident server. A product's threadgroups
walk `1 << swizzle` row tiles under each weight tile before moving on (`block.zig`). MLX's 2 sends a 2048 picture's
4,120 rows past every weight tile 16 times; 6 keeps all 65 row tiles on one weight tile while it is cached. Each
output's sums are unchanged. Step GPU seconds in one server per row, A and B alternated in the same process:

| swizzle | 2048 step | 4096 step 1 |
|---|---|---|
| 2 (before) | 1.90-1.91 | 10.29 |
| 4 | 1.84-1.85 | 10.22 |
| 5 | | 10.17 |
| 6 (now) | 1.79-1.81 | 10.14 |
| 7 | 1.91 | |
| 6 on a 128 x 128 tile | 1.84 | |

With 6, warm 2048 requests took 12.54 and 12.99 s (6 steps 10.81 and 10.93 s) at a load average near 30, against
episode 1's recorded 13.03 s (6 steps 11.39 s). Every latent and PNG is sha1 `f3563c56…` / `904814d6…`, the same bytes
as card 68's warm runs before the change. A TOP-menu voice note sent during step 2 of a later request came back whole
in 8.38 s; that step waited 7.31 s for speech and reported `timing_valid=false`.

### H3's own products, measured

Card 84 replaced MLX's `gemm_nax` with `h3_gemm_32x64` (`kernels/metal/h3/h3_gemm.metal`): 64 x 128 a threadgroup,
each of its four warps 32 x 64, so a loaded fragment feeds two MMAs where MLX's 32 x 32 warp feeds one. Each output's
K chain is MLX's, 32-deep steps in order. Ten dispatches a shape in one command buffer, 4,120 rows, best of three:

| Product | Shape (n x k) | MLX | H3's own |
|---|---|---|---|
| fc1 | 28,672 x 5,376 | 118 | 116 TFLOP/s |
| q k v | 21,504 x 5,376 | 118 | 116 |
| out | 5,376 x 7,168 | 113 | 108 |
| fc2 | 5,376 x 14,336 | 89 | 103 |

The products are at the rate MLX's kernel reaches and a wider tile does not move them (card 150's isolated replays
later put that rate at under two fifths of the tensor unit's own ceiling); the only shape that gains is fc2, worth
0.3 s of the six steps. Warm 2048 requests took 12.12 and 12.13 s (6 steps 10.49 and 10.52 s, decoder 1.38 and 1.40 s)
against 12.30 s before. The latent and PNG stay sha1 `f3563c56…` / `904814d6…`.

A profile stage's seconds cover all 300 block calls of the six steps, so attention is not a gap either: its 1.54 s is
149 TFLOP at 96 TFLOP/s, the products' own rate, and a step is 168 TFLOP. Six steps are 1,006 TFLOP in 10.7 s, about
94 TFLOP/s against the 118 the fc1 shape reaches.

### int8 fc1, landed

The same 32 x 64-warp loop on `int8_t` fragments with `int32` sums ran the fc1 shape (4,096 x 28,672 x 5,376) at
215 TOP/s against 96 TFLOP/s for bf16 in one probe; `half` ran at bf16's rate. Card 34's prompt, seed 42, one resident
server, a warm 2048 picture each, against episode 1's latent and PNG (card 80's cross-precision rule: cosine >= 0.99,
PSNR >= 28.2 dB, mean pixel diff <= 4.1):

| fc1 quantizer | latent cosine | PSNR | mean pixel diff | max | 6 steps | picture |
|---|---|---|---|---|---|---|
| both operands at a row's absolute max | 0.994759 | 28.342 | 4.365 | 243 | 9.24 s | 10.99 s |
| the same on fc1 and fc2 | 0.942927 | 18.733 | 14.849 | 255 | 9.23 s | 10.92 s |
| weights per 256 values of K, activations a row (landed) | 0.995344 | 28.240 | 4.011 | 245 | 9.54 s | 11.14 s |

A row's absolute max misses the mean floor by 0.27. fc2 in int8 does not hold the picture at all: SwiGLU's rows have
outliers that one row scale cannot cover. fc1's weights grouped per 256 values of K hold the rule, so fc1 runs in int8
(`h3_gemm_32x64_i8`): `bind` quantizes every block's fc1 rows once (`h3_quantize_rows`, one simdgroup a row, its
groups in turn) into a private 7.7 GiB copy, each block quantizes `xn` a row before its product, and each 256-deep
block's int32 sums fold into fp32 at the weights' scales for that block, the row's scale last. The fold costs about
0.10 s a step against the ungrouped quantizer. The other products stay bf16 and byte-identical (until card 136 put
them on the same kernel). Only the released layout's (a picture's) fc1 ran in int8: a FastH3 clip's binding makes no
copy and its linears stay the bf16 product, which `test-h3-gemm` holds bit for bit against MLX's kernel at whole and
ragged rows.

`tf-h3-image-server` joins each binding's int8 copy to its weights' residency set while the binding lives: in shared
storage a step's encode took 0.42 s, private 0.19 s, joined 0.04 s. Cold and warm requests give the same latent
(sha1 `8b9e75b5…`, PNG `7bf89a7f…`), and `test-h3-gemm` holds the quantizer and the int8 product against an exact
dequantized product to one bf16 rounding across a ragged tile.

Built from this card's head, the same server and request lines: warm 2048 pictures took 11.14 and 11.16 s (prompt 0.22, 6 steps
9.54 and 9.47, decoder 1.37 and 1.43), against 12.12 s in bf16 and episode 1's 13.03 s; the cold one 16.05 s with a 3.78 s bind. The
receipts are kept outside the repo in `h3-image-runs/84/rec/ship*`. The 11 s target is not met; the decision on this
card took the same picture at about 11.1-11.4 s over a changed one under 11 s.

### Card 124: the Spark's wins on the Metal path

Card 124 tried each Spark (CUDA) technique on this path. Same prompt, seed 42, six steps, one resident server; the
gates are card 80's (cosine >= 0.99, PSNR >= 28.2 dB, mean pixel diff <= 4.1) plus byte-identity where arithmetic is
unchanged. The stage table is `tf-h3-image-server profile` (the diagnostic mode: each stage waits, so its sum tracks
GPU work, not wall): card 84's column is the profile recorded when the switch landed (a warm request on that tree at
load 12, six steps summed); card 124's is the `runV`/`runW` profile on the shipped tree. Both columns are bf16
everywhere but fc1's int8 — the stage deltas are run-to-run spread on a loaded machine, not the change's effect.

| Stage | Card 84 (bf16 + int8 fc1) | Card 124 (shipped) |
|---|---|---|
| rows in | 0.02 s | 0.015 s |
| norms | 0.14 s | 0.137 s |
| q k v gate | 2.48 s | 2.436 s |
| heads | 0.10 s | 0.102 s |
| attention | 1.54 s | 1.529 s |
| attention out | 0.89 s | 0.886 s |
| mlp in | 2.40 s | 2.527 s |
| mlp out | 2.20 s | 1.820 s |
| final | 0.04 s | 0.012 s |

1. **Block-scaled low-precision projections.** int8 qkv at a row's activation scale: warm totals 10.51/10.68/10.95 s
   but PSNR 26.16, mean pixel diff 5.40, cosine 0.9915 — under the PSNR floor; reverted. The same product with
   activation scales per 256 values of K (a `sa_groups` variant, `h3_gemm_32x64_i8s`, folding each block's int32 sums
   at both operands' block scales): gates pass (cosine 0.9962, PSNR 29.81, mean 4.10) but warm totals 11.43/11.48/11.73
   s — qkv's product saved only ~0.17 s against ~1 s of extra quantize and fold work; reverted. Extending the grouped
   scheme to attention out and fc2 (all four projections int8): 11.74/12.85/13.32 s; reverted. On these shapes and
   this GPU the int8 product's win (~2.2x on a probe) is smaller than the grouped fold's cost; only fc1's ungrouped
   row scale still nets a saving (card 84).
2. **Fused attention.** Already fused: `h3_attention` is one online-softmax kernel; nothing to port. It runs at
   1.52-1.54 s of a ~9.5 s six-step total — about 96 TFLOP/s, the products' own rate, so a deeper pipeline is not the
   next saving either.
3. **One command buffer a step.** Tried as four blocks a buffer while the speech lock is free (a Spark-style larger
   submit without queueing the whole forward ahead of speech), a block its own buffer once speech takes the lock.
   Two findings: no wall gain (a step is GPU-bound at ~1.5 s; host encode already hides), and a latent change —
   `.concurrent` encoders do not order dispatches, and the blocks rely on a buffer boundary for the last-writer
   dependency into the next block's norm. Reverted; the per-block commit stands.
4. **Prompt-row work.** `h3_gate_add` and `h3_final_norm` skip the `media` un-stepped rows via buffer offsets (the
   kernels index `LINE`/`TIME` and `X`/`Y`/`normed` by their grid row, so offsetting every row-indexed buffer
   keeps the stepped rows bit for bit; prompt rows are re-copied into `x` and condition rows re-projected by
   `rows in` each step, and `normed` is read only by `rows out`). Latent sha1 `8b9e75b5…` before and after, warm
   totals 11.57-11.96 s against card 84's 11.14-11.16 — the ~0.6% of rows is below run noise.

The shipped tree's three warm runs, quoted (card 84's lines are `prompt 0.22, 6 steps 9.54/9.47, decoder 1.37/1.43`,
above): `runU: prompt 0.22 s, case 0.01 s, transformer reused 0.00 s, 6 steps 10.32 s, decoder 1.41 s, total 11.96
s`; `runV: prompt 0.22 s, case 0.01 s, transformer reused 0.00 s, 6 steps 10.03 s, decoder 1.36 s, total 11.62 s`;
`runW: prompt 0.22 s, case 0.01 s, transformer reused 0.00 s, 6 steps 9.98 s, decoder 1.35 s, total 11.57 s`.
Selftests pass (`zig build test test-golden`, h3 gemm/prompt/decoder selftests included).

The 10% bar (about 10.0 s) is not met on this silicon: the step is GPU-bound at the products' rate, the remaining
wall-GPU gap is ~0.1 s a step of host round-trips (latent up, velocity down, host Euler), and every int8 widening
either fails the picture or costs its win in the fold. The lever worth a next card is a GPU-side Euler with the
latent resident across steps (`h3_euler` exists on the CUDA path); that also deletes the per-step velocity
download.

### Card 135: the Spark's norm-quantize fusion on the Metal path

Card 128's lever 2 on the picture path's launches: `h3_norm_quant` runs the gated residual add, the modulated
RMSNorm and `h3_quantize_rows`' quantization as one launch — each lane recomputes the normed value for a block's
absmax and the rounding, so the bf16 normed rows are never written (the int8 fc1 product was their only reader;
norm1's bf16 `xn` still feeds the bf16 qkv product). Its scales land in `xs` at the gemm's own layout, one scale
per 256-value K block a row (21 a row at hidden 5,376): the card-136 merge brought the block-scaled gemm this
fusion predates, so the grouped form is what it must feed. A block's launched passes go 10 to 9; a clip's bf16
path is unchanged.

`test-h3-gemm` holds the fusion byte for byte against the two launches it replaces at `g = 256`, and the contract
runner's `h3_norm_quant` entry byte-compares it to `h3_norm` then `h3_quantize_rows` at that grouping on the same
inputs. A resident server's warm 2048² latent and PNG are sha1-identical to this main's (`776b443a…`, `3cda60cf…`),
with the step at 1.53 s of GPU against main's 1.54 s and the warm request at 10.6-11.6 s against 11.1 s. An
activation-in-epilogue variant of fc1's int8 product (`silu(gate)·value` folded in, `h3_swiglu` and the `wide`
rows dropped) was built and measured before review: a first layout keeping both K halves' fp32 accumulators
live ran the step at 1.73 s GPU, and a second shape reusing one accumulator over sequential value and gate
sweeps ran 1.63 s — a net loss, so fc1 keeps `h3_gemm_32x64_i8` + `h3_swiglu`.

### Card 136: int8 on all four linears

Every picture linear — the fused qkv projection, attention out, fc1 and fc2 — runs on `h3_gemm_32x64_i8`, now with a
scale per 256 values of K on **both** operands: card 124's measurements showed a row's activation scale breaks the
picture (26.16 dB on qkv) while the grouped fold held the gates, and the same applies to SwiGLU's outliers on fc2's
input. `bind` quantizes all four projections of every block once into private `wq`/`ws` (~20 GiB of int8 weights for
the release checkpoint, joined to the weights' residency set as the fc1 copy was); each product quantizes its input
into the shared `xq`/`xs` scratch, ordered by the same stage barriers that already bound the block. A clip's FastH3
blocks make no copy and keep their bf16 products.

A/B on this Mac, card 34's prompt, seed 42, warm 2048 requests on one resident server each (the latents below are
the second of two warm requests; every run's latent was byte-identical within its own build):

| | latent sha1 | cosine vs main | PSNR | mean diff | steps | total |
|---|---|---|---|---|---|---|
| main | `8b9e75b5…` (the pinned value) | — | — | — | 9.63/9.64 s | 12.50/13.27 s |
| int8 all four | `776b443a…` | 0.996884 | 30.613 dB | 3.451 | 9.33 s | 11.18 s |

The picture holds card 80's rule with more headroom than fc1's row-scaled landing (28.240 dB): the grouped fold pays
its accuracy back. The steps run ~0.3 s faster than this main build's, where card 124's all-four trial measured
slower (11.74-13.32 s totals) — this kernel takes one activation scale per lane row per K block, beside the column
scales it already read. The bind pays ~5 s once (20 GiB to quantize against 7.7); a resident server amortizes it.
`zig build test test-golden` is green apart from `message_starts_proof_test`'s timing bound, which fails identically
on unmodified main; the contract's int8 product serves all four projection shapes against float64 (worst 0.995 of
the bound).

### Card 137: the finish pass and the Euler on the GPU

Card 124's last lever landed: the step's tail is one fused `h3_head_norm` dispatch (final layer's modulated norm
and the step's head, the normed row in threadgroup memory, `normed` buffer and `h3_final_norm`/`h3_rows_out`
gone) and `h3_euler` steps `x += delta * v` on the resident rows in `picture.euler`'s two roundings — its product
goes through a volatile because the Metal compiler contracts to fma and skips the first rounding. The video rows
stay on the GPU across all six steps: one upload, one download, no per-step velocity readback. The decoder's pass
batch went to 128 tiles so 2048's 121 decode in one pass (card 130's batch; its `h3d_gemm` epilogue had already
folded the bias and LayerScale).

Gates: the warm 2048 latent is **bit-identical** to main's (sha1 `8b9e75b5…` matches the card 124 receipt) and the
PNG matches byte for byte (`7bf89a7f…`). Warm A/B, ReleaseFast server, the base the card's merge base `561d85c1`
(before card 136's int8 linears landed on main; the finish pass is orthogonal to them), alternating rounds of one
cold then three warm requests each, load average 24-40: head warm 11.13-11.74 s (decoder 1.38-1.61 s), base warm
11.06-11.79 s (decoder 1.36-1.48 s); two base runs were spoiled by contention and excluded. On wall clock the
change is flat — the ~0.1 s a step of host round-trips is small next to a 1.6 s step — while the GPU-pass meter
shows the decoder's 121 tiles at 1.08 s in one pass against 1.36 s over two, and the tail kernels' launches replace
two dispatches and a buffer. The win is structural, not a new warm number.

### Card 150: the int8 product's rate, from isolated kernel replays

The card asked for one warm 2048 picture profiled per kernel with Metal GPU counters or a capture. No in-picture
Metal counter profile was taken: no capture or counter sample exists, and the Desk closed the Mac's GPU windows
before one could be; the in-picture profile moves to card 148. The table below is not a picture profile. Each row is
one product launch replayed alone (`tools/zig/metal_bench.swift`, command-buffer GPU timestamps, warm medians of 12,
the exact launch `product()` dispatches at 4,120 rows; every bench under `/tmp/tensorfold-model.lock`, load average
beside each set). A replay has no neighbouring stages or barriers around it, so its times bound the kernel in
isolation; they are not in-picture kernel times and not a share of a picture's 9.24 s of step GPU time:

| `h3_gemm_32x64_i8` launch | Shape M x N x K | time | rate |
|---|---|---:|---:|
| qkv | 4,120 x 21,504 x 5,376 | 6.88 ms | 138 TOPS |
| attention out | 4,120 x 5,376 x 7,168 | 2.54 ms | 125 TOPS |
| fc1 (gate+value) | 4,120 x 28,672 x 5,376 | 9.13 ms | 139 TOPS |
| fc2 | 4,120 x 5,376 x 14,336 | 5.22 ms | 122 TOPS |

(First set at load 47-92, the rerun at load 4.5 within 3% of these.) The peaks the products run against, measured
the same way with register-fed `mma_16x32` chains — 16,384 simdgroups of 80,000 chained ops through MPP's
`matmul2d` cooperative-tensor op, the M5 tensor-unit path every product kernel already takes, its
cooperative-tensor staging included: **493 TOPS int8 and 255 TFLOPS bf16**. Replayed alone, the int8 product sits at
25-28% of its path's own ceiling, and the bf16 products (the same probe's 255 TFLOPS against their measured 96) at
38%: the operand gather, not the tensor unit, is what binds — each fragment loads as eight scalar reads.

Candidates, each microbenched against the shipped kernel first (medians; both sets, loads 55-80 and 4.5):

| Candidate | qkv | out | fc1 | fc2 | Verdict |
|---|---:|---:|---:|---:|---|
| shipped (byte-wise `frag_get`) | 6.88 ms | 2.54 ms | 9.13 ms | 5.22 ms | kept |
| int8 fragments as two 4-byte loads (`frag_get_q`) | 6.01 ms | 2.09 ms | 7.98 ms | 4.23 ms | -12 to -19% isolated; **lost the served A/B, reverted** |
| 64 x 256 tile, 8 warps a threadgroup | 7.56 ms | 2.73 ms | 9.85 ms | 5.65 ms | slower by 8-10%: A-fragment traffic was not what binds |
| register double-buffer of the 32-deep steps | 11.90 ms | 4.11 ms | 15.88 ms | 8.33 ms | slower by ~70%: spills past the register budget the two accumulator sets fill |
| scale fold removed (int32 accumulator through all of K) | 11.62 ms | 3.98 ms | 15.41 ms | 8.16 ms | slower by ~70%: the per-256 fold also breaks the mma accumulator chain each block — it is structure, not just cost |

SwiGLU folded into fc1's product stayed unbuilt: card 135's folded variants measured 1.63-1.73 s a step against
1.53 s for the pair, in exactly the two register budgets the double-buffer row above re-measured as spills (both K
halves' accumulators live, or one accumulator swept sequentially). A folded try must beat fc1's 9.13 ms plus
`h3_swiglu` without both accumulator sets.

The 4-byte-load candidate is the card's cautionary receipt. It was gated before it was benched: the warm latent was
byte-identical (`776b443a…`, the pinned value), card 80's rule passed against episode 1's Mac latent and PNG
(`h3-image-runs/34/t2i-2048.latent.f32`, `fox-2048-seed42.png`: cosine 0.995459, PSNR 28.673 dB, mean pixel diff
3.903, every step `timing_valid=true`), `test-h3-gemm`, `test-h3-image-server`, `test-h3-image-prompt` and the
kernel contract's `h3 product` set (gates 1-4, 0 limit problems) were green. The decoder's products are `h3d_gemm`,
not `h3_gemm_32x64_i8`, so its same-latent bound holds by construction: the candidate's byte-identical latent is the
decoder's same input (the candidate's PNG hash itself was not kept). `tf-h3-image-decoder selftest` on this M5 Ultra
(one lock hold, 40 s, load 7.5) passed every bound: 24 blocks relative 1.49e-2, worst 16.5 ULP, picture bytes at
most 1 apart, 121 tiles in passes of 128 bit-identical to passes of 16. But the served A/B — two resident
ReleaseFast servers, one a side, cold on card 34's prompt then three warm runs on unseen prompts (harbour, meadow,
steam; `docs/recipes/minimax-h3-cuda/prompts/`), one request a lock hold, seed 42, six steps — reversed it:

| Warm run (unseen prompt) | head 6 steps | head total | main 6 steps | main total | load avg (head/main) |
|---|---:|---:|---:|---:|---|
| 1 | 9.15 s | 11.44 s | 8.94 s | 11.22 s | 12.2 / 8.9 |
| 2 | 9.67 s | 12.36 s | 9.25 s | 11.51 s | 10.8 / 8.0 |
| 3 | 9.67 s | 11.27 s | 9.24 s | 10.83 s | 10.0 / 7.5 |

Every paired warm run is main's. An earlier round under worse contention (main's requests at load 31-52 with the
golden suite building, head's at 15-48) measured head 9.20-9.82 s against main 10.44-11.72 s and decoder walls up to
6.5 s — load, not the kernels, moved those numbers, which is why the quiet round above is the one that decides. The
isolated win (182 ms a step of product time, reproduced at both loads) did not appear in the served pipeline, which
came out 0.21-0.43 s slower in the six steps and 0.22-0.85 s slower in total a picture; the microbench cannot see
the step's barrier-ordered stages around each product, and nothing within this card isolated which of them the
wider loads perturbed.
By the rule — kept only if faster — the code was reverted; the receipts stand so the next round does not re-try the
widened loads blind. Fast mode (`--draft-steps 3`) on the shipped state, same protocol at load 25-30: warm 8.59 s
and 6.17 s (the second with the transformer binding reused), totals 11.19/7.86 s; one warm run waited up to 0.52 s
for speech (`timing_valid=false`) and is reported, not counted.

The Desk's GPU window (16:37-17:17 HKT, voice idle) re-ran the A/B with `frag_get_q` rebuilt on this branch's
parent (`fc8aa2d45`, the shipped side built from the same commit): one resident ReleaseFast server a side, one lock
hold a server (cold on card 34's prompt, then harbour, meadow, steam at seed 42; latents byte-identical between
sides on all three prompts), sides alternated main, head, head, main, main, head. Load is the 1-minute average when
each request was sent; other checkouts' builds kept it at 8.8-45.0. Every warm run's stage times in seconds (`b:`
bound, `r:` reused; step GPU sum in brackets):

| Round | Side | Request | prompt | case | xform | 6 steps (GPU) | decoder | total | load | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | main | harbour | 1.28 | 0.01 | b:0.75 | 8.97 (8.94) | 1.41 | 12.42 | 23.2 | |
| 1 | main | meadow | 1.24 | 0.00 | b:0.76 | 9.28 (9.24) | 1.38 | 12.66 | 20.9 | |
| 1 | main | steam | 1.69 | 0.01 | r:0.00 | 9.31 (9.24) | 1.39 | 12.40 | 18.8 | |
| 1 | head | harbour | 1.52 | 0.01 | b:0.81 | 8.96 (8.93) | 1.46 | 12.76 | 33.0 | |
| 1 | head | meadow | 1.02 | 0.01 | b:0.76 | 9.28 (9.24) | 1.46 | 12.52 | 29.4 | |
| 1 | head | steam | 0.76 | 0.01 | r:0.00 | 9.95 (9.92) | 1.60 | 12.32 | 26.9 | |
| 2 | head | harbour | 2.03 | 0.06 | b:1.30 | 9.81 (9.02) | 3.23 | 16.43 | 19.0 | |
| 2 | head | meadow | 3.07 | 0.01 | b:1.20 | 10.39 (9.36) | 2.72 | 17.38 | 15.2 | |
| 2 | head | steam | 2.73 | 0.09 | r:0.00 | 9.61 (9.32) | 3.83 | 16.26 | 14.3 | |
| 2 | main | harbour | 2.32 | 0.01 | b:1.46 | 10.81 (9.03) | 3.31 | 17.92 | 15.2 | |
| 2 | main | meadow | 23.90 | 0.02 | b:1.96 | 10.61 (9.37) | 2.61 | 39.09 | 25.3 | prompt waited 20.78 s for speech |
| 2 | main | steam | 2.23 | 0.06 | r:0.00 | 9.60 (9.23) | 2.90 | 14.79 | 16.5 | |
| 3 | main | harbour | 2.28 | 0.01 | b:1.10 | 9.00 (8.94) | 1.65 | 14.05 | 28.0 | |
| 3 | main | meadow | 1.44 | 0.03 | b:0.91 | 9.31 (9.24) | 1.98 | 13.67 | 25.2 | |
| 3 | main | steam | 2.69 | 0.05 | r:0.00 | 9.72 (9.24) | 2.45 | 14.90 | 22.4 | |
| 3 | head | harbour | 2.20 | 0.01 | b:1.18 | 10.24 (9.01) | 3.40 | 17.03 | 21.9 | |
| 3 | head | meadow | 3.32 | 0.01 | b:1.08 | 9.41 (9.24) | 1.97 | 15.80 | 24.9 | |
| 3 | head | steam | 2.70 | 0.01 | r:0.00 | 9.28 (9.24) | 2.26 | 14.25 | 19.0 | |

Medians of the nine warm runs a side: main 9.31 s wall / 9.24 s GPU of steps, head 9.61 s wall / 9.24 s GPU. Every
warm step reported `timing_valid=true`; round 2 main's meadow prompt waited 20.78 s for speech, which spoils its
total but not its steps (main's median stays 9.31 s without it), and head's round-1 cold run (not in the table)
waited 0.31 s in prompt and 0.53 s in steps 1-2 (`timing_valid=false`).

Fast mode (`--draft-steps 3`), same stage breakdown:

| Round | Side | Request | prompt | case | xform | 6 steps (GPU) | decoder | total | load | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | main | harbour | 2.16 | 0.02 | b:1.23 | 8.98 (5.88) | 2.13 | 14.52 | 19.4 | |
| 1 | main | meadow | 3.37 | 0.02 | b:0.94 | 8.65 (6.16) | 1.83 | 14.82 | 18.3 | 0.17 s speech wait; steps 1, 5-6 `timing_valid=false` |
| 1 | main | steam | 2.68 | 0.02 | r:0.00 | 6.16 (6.09) | 2.13 | 11.00 | 15.9 | steps 4-6 `timing_valid=false` |
| 1 | head | harbour | 4.19 | 0.02 | b:1.21 | 10.98 (5.88) | 3.03 | 19.43 | 11.3 | |
| 1 | head | meadow | 3.57 | 0.02 | b:1.19 | 9.49 (6.16) | 2.43 | 16.70 | 8.8 | |
| 1 | head | steam | 2.79 | 0.02 | r:0.00 | 6.49 (6.09) | 2.90 | 12.20 | 9.9 | |
| 2 | main | harbour | 25.82 | 0.01 | b:1.46 | 9.51 (5.88) | 2.91 | 39.71 | 22.4 | prompt waited 23.12 s for speech |
| 2 | main | meadow | 4.21 | 0.06 | b:1.33 | 9.51 (6.16) | 2.50 | 17.61 | 20.9 | |
| 2 | main | steam | 2.02 | 0.01 | r:0.00 | 6.67 (6.09) | 2.92 | 11.62 | 15.6 | |
| 2 | head | harbour | 2.58 | 0.02 | b:1.30 | 8.88 (5.87) | 3.00 | 15.78 | 11.5 | |
| 2 | head | meadow | 2.21 | 0.01 | b:1.39 | 9.37 (6.15) | 8.22 | 21.20 | 15.8 | |
| 2 | head | steam | 4.49 | 0.03 | r:0.00 | 7.30 (6.16) | 8.29 | 20.12 | 45.0 | |

Three of main's fast-mode runs were spoiled: round 1 meadow (load 18.3) and steam (load 15.9) had steps with
`timing_valid=false`, and round 2 harbour's prompt (load 22.4) waited 23.12 s for speech. They are reported above but
not counted, and they were never rerun here: the Desk closed the window and the lead ruled out further GPU windows for
this card, so clean fast-mode reruns move to card 148. The fast-mode comparison is therefore unbalanced and is not a
verdict; the verdict rests on the nine-a-side six-step A/B above. Over the clean runs only: main 8.98 s wall / 6.09 s
GPU of steps (three runs), head 9.12 s wall / 6.12 s GPU (six runs; round 1 head's cold run waited 0.31 s for speech
in its prompt). Matched to main's three clean requests (round 1 harbour, round 2 meadow and steam), head's medians
are 9.37 s wall / 6.15 s GPU against main's 8.98 / 6.09 s. No fast-mode win or loss is claimed from these.

The window confirms the quiet round at six steps: even on pure step GPU time (9.24 vs 9.24 s, nine runs a side) the
4-byte loads yield no served gain, so the code stays reverted. Neither side reached the quiet baseline's 10.8 s
normal / 7.8 s fast totals under this host load (8.8-45.0); the best warm totals in the window were main 12.40 s and
head 12.32 s at six steps, and main 11.00 s (`timing_valid=false`; 11.62 s valid) and head 12.20 s in fast mode.

### Card 184: the int8 weights packed in fragment order, kept

Card 150's table puts the int8 product at 25-28% of its path's ceiling, each B fragment gathered as eight scalar
reads. Bind now quantizes every block's four projections straight into the order the MMA's right operand takes them
(`h3_packed` in `h3_gemm.metal`, `packedAt` in `kernels.zig`): each 32-row band, then each 16-deep K step, holds 32
lanes of 16 bytes, a lane's two 16 x 16 fragments for one `mma_16x32`. `h3_gemm_32x64_i8` loads them as one 16-byte
read where main made eight; A, the fold order and every value are main's. `h3_quantize_rows` takes a fourth parameter
(nonzero writes packed); the step's own input quantizers stay row-major.

Isolated replays, the same harness as card 150 (`tools/zig/metal_bench.swift`, main's kernel against this card's on the
same int8 operands, head's B packed by `packedAt`), warm medians of eight command buffers in µs, three sets a side
under `/tmp/tensorfold-model.lock`; every output byte-identical to main's on all four shapes:

| `h3_gemm_32x64_i8` launch | Shape M x N x K | Card 150 | main, set 1 | head, set 1 | main, set 2 | head, set 2 |
|---|---|---:|---:|---:|---:|---:|
| qkv | 4,120 x 21,504 x 5,376 | 6.88 ms | 9.16 ms | 5.82 ms (-36%) | 17.45 ms | 13.63 ms (-22%) |
| attention out | 4,120 x 5,376 x 7,168 | 2.54 ms | 3.15 ms | 2.81 ms (-11%) | 6.45 ms | 4.97 ms (-23%) |
| fc1 (gate+value) | 4,120 x 28,672 x 5,376 | 9.13 ms | 15.35 ms | 7.40 ms (-52%) | 22.88 ms | 17.81 ms (-22%) |
| fc2 | 4,120 x 5,376 x 14,336 | 5.22 ms | 5.27 ms | 4.10 ms (-22%) | 13.00 ms | 10.04 ms (-23%) |

Set 1 ran at load 92-95, set 2 at 26-41 while another card's resident server shared the GPU, so neither is a quiet
rate; within a set, head beats main on every shape and every one of the three runs.

Served warm A/B, ReleaseFast `tf-h3-image-server`, main `bc1012c35` against this card's head, one resident server a
session holding the lock for its own 39-68 s: card 34's pilot cold, then harbour, meadow and steam warm and unseen (seed
42, 2048x2048). Every step of all five sessions is `timing_valid=true`, and every latent and PNG is sha1-identical
between the builds: the pilot's latent is `776b443a…` (card 135's pin) and its PNG `85098fa9…`. Seconds of the six
steps, cold then the three warm (total in brackets):

| Session | Load | pilot (cold) | harbour | meadow | steam |
|---|---|---|---|---|---|
| head | 39.0 → 28.4 | 8.22 (10.23) | 7.60 (9.37) | 7.70 (9.54) | 7.69 (8.53) |
| main | 17.9 → 12.8 | 9.25 (11.29) | 8.94 (10.51) | 9.25 (10.81) | 9.22 (10.07) |
| head | 12.1 → 9.8 | 7.68 (9.48) | 7.56 (9.33) | 7.70 (9.51) | 7.68 (8.55) |
| main | 13.1 → 13.2 | 10.70 (13.36) | 9.08 (10.89) | 10.63 (13.00) | 9.27 (10.23) |
| head | 13.1 → 16.9 | 8.04 (10.29) | 8.72 (10.85) | 8.77 (11.18) | 7.75 (8.74) |

Head's warm six steps are 7.56-8.77 s against main's 8.94-10.70 s here and card 148's 9.20 s, about 1.5 s off a
picture at the quietest pair. The packed byte writes cost bind 0.2 s (0.91-1.34 s against 0.72-1.25 s), which a
warm picture that rebinds still more than earns back. The load never fell under the 8 the card asked for: the
window's other builds and services kept it at 10-39, so these are the best this window gave, not quiet numbers.
Receipts (`bench.py`, `bench-load92.log`, `bench.log`, `session.sh`, `contract.log`) are in `h3-image-runs/184`,
outside the repo.

### Card 151: a faster `h3_attention`, measured and reverted

`h3_attention` already runs q·kᵀ and p·v on the tensor units (`matmul2d`, 8 simdgroups, 64-row query tiles) with an
fp32 online softmax. A throwaway microbench ran it at the warm 2048 fox geometry (56 heads, 65 key tiles, 4,122 rows,
0.50 TFLOP a call): **5.27-5.29 ms, 94 TFLOP/s**. Times 300 block calls that is 1.58 s, matching card 124's profile
stage (1.53 s of ~9.5 s, about 16% of the six steps). Two variants under `/tmp/tensorfold-model.lock`, load 7-9:

| Variant | Best | TFLOP/s | Output vs current |
|---|---|---|---|
| current: 3 barriers a key tile | 5.27-5.29 ms | 94 | — |
| row max and sum by `simd_shuffle_xor` (the row's 4 threads are lane neighbours): 2 barriers, no `tops` tile | 4.83-4.89 ms | 102 | max relative 7e-5 |
| 32-row tiles on 16x32 simdgroup fragments, 2 simdgroups (the Spark kernel's shape) | 21.5 ms | 23 | abandoned: 4x slower |

The shuffle variant, end to end: warm 2048 on a resident `tf-h3-image-server`, seed 42, two prompts not used before
(an origami crane, a teapot), one cold and three warm requests each, against current main's build in the same
session. Each server held the lock for its own lifetime. Load average is shown before → after each request:

| Build, prompt | Warm request (6 steps, decoder, total) | Load |
|---|---|---|
| main, crane | 9.23 / 1.44 / 10.67 s; 9.21 / 1.46 / 10.68 s; 9.23 / 1.45 / 10.68 s | 6.0 → 7.8 |
| shuffles, crane | 9.20 / 1.46 / 10.66 s; 9.22 / 1.45 / 10.68 s; 9.20 / 1.45 / 10.65 s | 8.6 → 10.9 |
| main, teapot | 9.26 / 1.37 / 10.64 s; 9.21 / 1.39 / 10.61 s; 9.21 / 1.37 / 10.58 s | 9.5 → 6.4 |
| shuffles, teapot | 9.24 / 1.36 / 10.62 s; 9.18 / 1.36 / 10.55 s; 9.18 / 1.36 / 10.55 s | 12.1 → 10.0 |
| main, crane, `--draft-steps 3` | 6.08 / 1.40 / 7.49 s; 6.09 / 1.38 / 7.48 s; 6.14 / 1.39 / 7.55 s | 5.6 → 4.2 |
| shuffles, crane, `--draft-steps 3` | 6.15 / 1.39 / 7.55 s; 6.09 / 1.38 / 7.48 s; 6.08 / 1.39 / 7.49 s | 4.2 → 4.0 |

The colds were 11.86-11.96 s (main) and 11.62-11.88 s (shuffles). Contention spoiled the first teapot pair (main's
steps 9.36-9.63 s with decoder 2.0 s; the shuffles' cold 18.2 s). Both builds were rerun, and the spoiled logs sit next
to the receipts as `*.spoiled.server.log`.

**Not kept.** The 0.4 ms a call measured alone would save ~0.13 s a picture, but warm steps differ by at most 0.04 s.
The rows' max and sum are reduced in a different order, so the picture also changes: against main's warm latent and
PNG on the same prompt, the crane has cosine 0.9923 with mean pixel diff 5.17. The teapot has cosine **0.9889** with mean
pixel diff 6.38, under card 80's 0.99 and over its 4.1. Six steps amplify the 7e-5 per-call difference. Larger
query tiles were not tried: a 64 x 64 tile's fp32 scores and bf16 probabilities take 24 KiB of threadgroup memory,
and 128 query rows would need 48 KiB against the 32 KiB limit. The kernel is unchanged from main. The receipts
(`ab151.sh`, server logs, latents and PNGs) are in `h3-image-runs/151`, outside the repo.

### Card 162: scores in registers and 128-row query tiles, measured and reverted

I kept `h3_attention`'s q·kᵀ scores in the product's cooperative tensor instead of the 16 KiB fp32 `st` tile, so 128
query rows fit: 18,944 threadgroup bytes instead of 25,856 for 64 rows. A layout probe on the M5
(`matmul2d_descriptor(64, 64, 128)`, 8 simdgroups) gives each lane 4-column quarters of 16-column row chunks. A
chunk's four quarters sit on lanes `l`, `l^1`, `l^8` and `l^9`. Its running sum is passed through those lanes by
`simd_shuffle_xor` (^1, ^9, ^1, ^9) in column order. Every fp32 sum then rounds exactly as main's quarter-row thread
rounds it, and row maxima go through a 1 KiB `part` tile. That leaves two barriers per key tile instead of four.
Unlike card 151's shuffles, the reduction order is main's.

Microbench on main's 2048 geometry (56 heads, 65 key tiles, 4,122 rows), built with the engine's compile options
(safe math, fast functions, language 4.1). These are best-of-30 times under `/tmp/tensorfold-model.lock`, two runs,
load 41-47 from other builders' compiles:

| Kernel | Threadgroup bytes | Time | Output vs main |
|---|---:|---:|---|
| main | 25,856 | 5.01-5.01 ms | — |
| scores in registers, 64-row tiles | 9,472 | 4.89-4.90 ms | 29,546,496 of 29,546,496 bf16 values bit-equal |
| scores in registers, 128-row tiles | 18,944 | 5.95-5.95 ms | bit-equal |

128 rows lose by 19%: there are half as many threadgroups (33 x 56) with twice the registers each. With exp inside
the per-role branch, the 64-row tiles ran 5.36 ms. Moving it ahead of the chain and selecting the sum gave 4.89 ms.
Built with Metal's default fast math instead, the two forms differed from main in 53 and 1,136 values by one bf16 step:
fast math contracts `acc * scale - next`, which main's `st` round trip prevents. With the engine's options it passed
`h3-image-contract` (32 of 32 kernels, 36 of 36 controls red, no limit problems).

End to end at 64-row tiles: warm 2048 on a resident `tf-h3-image-server`, seed 42, two prompts not used before (a
lighthouse, a cello), one cold and three warm requests each, against current main's build (`337abc0a3`) in the same
session. Each server held `/tmp/tensorfold-model.lock` for its own run. Load is shown before → after each request:

| Build, prompt | Warm request (6 steps, decoder, total) | Load |
|---|---|---|
| main, lighthouse | 9.25 / 0.78 / 10.04 s; 9.30 / 0.79 / 10.10 s; 9.24 / 0.77 / 10.01 s | 13.5 → 12.0 |
| registers, lighthouse | 9.37 / 0.78 / 10.16 s; 10.18 / 0.79 / 11.22 s; 9.23 / 0.81 / 10.06 s | 13.6 → 17.1 |
| main, cello (rerun) | 10.08 / 1.11 / 11.24 s; spoiled (197 s, see below); 9.37 / 0.80 / 10.18 s | 18.4 → 30.7 |

Every warm latent and PNG is byte-identical to main's on the same prompt and run (`cmp`, lighthouse and cello runs
2-4), so the picture gates hold by identity. Another builder's GPU job (`tf-exl3-encode`, outside the lock) spoiled
the first cello pair: main 10.31-10.87 s and the candidate 9.39-11.46 s in six steps. Both were renamed
`*.spoiled.server.log`. On the rerun, main's second warm request took 197 s; the Desk paused the GPU for the voice
server, and no further GPU run was started. So the candidate's cello rerun and the `--draft-steps 3` fast mode were
not measured.

**Not kept.** The 0.12 ms saved per call is ~36 ms a picture (300 block calls), under 0.4%. The one clean pair
(lighthouse) has warm steps 9.23-9.37 s against main's 9.24-9.30 s, with no win visible. The kernel would also depend on
the M5 cooperative layout above, which is undocumented, where main's `st` round trip is layout-agnostic. By the rule
(kept only if faster), I reverted the code. The variant (`h3_attention.r64.metal`), the layout probe, the
microbench (`bench.swift`, `cand.metal`), `ab162.sh`, the server logs, latents and PNGs are in `h3-image-runs/162`,
outside the repo.

### Card 169: the int8 product's operands through threadgroup memory, measured and reverted

Card 150 found the int8 product's operand gather binding at 25-28% of the tensor units' int8 peak. The candidate
(`f9d131dc3`) kept `h3_gemm_32x64_i8`'s tiles, warps, MMA chain and per-256 fold, but staged each K step through
threadgroup memory. The threadgroup's 128 threads load A's 64 rows and B's 128 rows as one 16-byte device read
apiece a row half (wide and coalesced). Each warp's fragments are then two 4-byte threadgroup reads, with rows
48 bytes apart so a fragment's lanes hit distinct banks. Stages are double-buffered: the next step's reads wait in
three `int4` registers a thread while the current stage multiplies, with one barrier a step. That is 18 KiB of
threadgroup memory, and the register cost is 12 words a thread, not card 150's second fragment set. The fold order
is unchanged, so every replay's output was byte-identical to main's on all four shapes.

Replayed alone with card 150's `metal_bench` launch: 4,120 rows, the `product()` grid, random int8 operands and
scales. Each set ran main and candidate interleaved three times in one lock hold (medians of the warm reps). This
Mac was not quiet: other checkouts' builds and GPU tests held load at 75-128 through every all-shape set, and main's
own replays ran 0-3x above card 150's 6.88/2.54/9.13/5.22 ms. Medians of the nine paired replays a side (sets at
loads 75-81, 96-97 and 124-128):

| Candidate | qkv | out | fc1 | fc2 | Verdict |
|---|---:|---:|---:|---:|---|
| main (same sets) | 8.32 ms | 3.21 ms | 9.75 ms | 7.79 ms | kept |
| staged, 32-deep steps, pitch 48 | 8.00 ms | 3.47 ms | 10.75 ms | 6.97 ms | ties: wins two shapes, loses two |

In the pairs where main ran within 4% of card 150's times, the two ran even: qkv 6.93 vs 6.86 ms, 6.84 vs
7.01 ms; out 2.62 vs 2.41 ms; fc1 9.16 vs 9.10 ms, 9.41 vs 9.23 ms. Variants, all measured on attention out against
main in the same lock hold:

| Variant | out, candidate / main | Note |
|---|---|---|
| 32-deep, rows 32 bytes apart (bank conflicts) | 5.77 / 6.47 ms | no worse than pitch 48: banks do not bind |
| 32-deep, rows 64 bytes apart | 6.19 / 6.47 ms | |
| 64-deep steps, 30 KiB, six `int4` a thread | 4.51-9.88 / 2.94-6.59 ms | 37-53% slower across three passes |
| 128-deep steps | — | 54 KiB of threadgroup memory, over the 32 KiB limit |
| `matmul2d` 32x64x32 straight from the staged tile | 7.87-7.93 / 6.39-6.42 ms | 23% slower: the op's own threadgroup tensor loads |

Staged or not, the fragments still reach the tensor unit through the same eight-element cooperative-tensor copies,
and wide device reads do not change that. The candidate pays for that copy in barriers and threadgroup round trips
instead of in scalar device reads, so it lands even. No candidate showed an isolated gain to carry into the served
A/B. Under this load an A/B would also measure the neighbours (card 150's quiet round moved 0.2-0.4 s with load
alone). So the served A/B was not run. By the rule (kept only if faster), the code was reverted. The kernel is
main's.

On the candidate commit, `test-h3-gemm` was green (28 s). The kernel contract's `h3 product` set passed gates 1-4 on
all four kernels (63 s, load 51-66), with one limit problem: the registry declares `h3_gemm_32x64_i8` at 0
threadgroup bytes against the candidate's 18,432, a declaration a kept kernel would have had to raise. The receipts
(`bench.py`, `spec.json`, `bench.log` with every set's timings and loads, the candidate and every variant's source)
are in `h3-image-runs/169`, outside the repo.

### Card 152: the decoder on the tensor units

A per-kernel trace of main's warm 2048 decode (`fc8aa2d45`, one dispatch a command buffer timed to completion, a
throwaway patch not committed; its total runs over the one-buffer pass) put the 1.4 s here:

| Kernel | main |
|---|---|
| `h3d_gemm` (qkv, out, w1, w2 of 24 blocks, `proj_out`) | 1,046 ms |
| `h3d_attention` | 201 ms |
| `h3d_post_quant` | 51 ms |
| `h3d_rms_norm` | 42 ms |
| `h3d_swiglu` | 40 ms |
| `h3d_heads` | 34 ms |
| `h3d_embed`, `h3d_layer_norm` | 2 ms |

The products are three quarters of it. Each of the acceptance's four levers, with its result:

- **Wider loads (landed).** The gemm runs on `nax.h`'s fragments (`mma_16x32`, bf16 operands straight from device
  memory, float32 sums, no threadgroup staging), and `h3d_attention` reads k four wide and v two wide. In the
  one-buffer pass, the GPU meter falls from 1.08 s to 0.56 s.
- **Fewer launches (landed).** SwiGLU moved into w1's product. Its mode 2 sums the gate and value rows of one output
  side by side, applies `bf16(bf16(SiLU(gate)) * value)` in the epilogue, and writes 4,096 values a row instead of
  8,192. A pass encodes 196 dispatches instead of main's 220 (embed 2, 24 blocks of 8 instead of 9, tail 2), and the
  2n-wide rows no longer go through memory. Episode 1's latent decodes to the same PNG bytes as before the fusion
  (`85098fa9…` in both builds), and the GPU meter falls from 0.56 s to 0.49 s.
- **FP16 products with FP32 sums (measured, not kept).** The fragments typed `frag<half>` convert every bf16
  activation and weight on load. Episode 1's latent decoded in 0.68 s of GPU twice, against bf16's 0.54-0.55 s,
  with the two builds alternating under the lock at load 21 and each build's runs byte-identical. The acceptance
  keeps a change only if it is faster, so the products stay bf16: bf16 operands give exact products in the
  fragments' float32 sums at the same rounding points.
- **PNG write overlapped (tried, not shipped).** After the pass, unpatchify, the blend and `picture_writer`'s PNG
  take 0.25 s at 2048, nearly all of it `png.encode`'s single zlib stream through `std.compress.flate`. Parallel
  bands, or another level or filter, change the compressed bytes. A band-by-band encode feeding one `flate.Compress`
  while the GPU decodes the next tile rows keeps them, and a split decode is bit-identical on this card's kernels.
  It was built and timed only in the standalone tool, never in a locked warm server A/B, so it was reverted; card 170
  below ships it with that A/B.

Gates: `tf-h3-image-contract` on the M5 runs the tensor-unit build (`h3d_gemm`'s SwiGLU form beside `h3d_swiglu`,
which every build still compiles) and that build's controls; a GPU without tensor units runs main's build and main's
controls. On the M5 Ultra (21 Nov, load 26-30): 32 kernels pass gates 1-4, all 36 controls go red as planted, and
there are no limit problems;
`tf-h3-image-decoder selftest` passes (24 blocks relative 1.48e-2, picture bytes at most 1 apart); `zig build test
test-golden` exits 0. Episode 1's latent decoded by this card versus main: 56.6 dB, mean 0.14, max 5. The warm
latents are bit-identical to main's on every request below. The cold picture against episode 1's Mac PNG
(`904814d6…`): latent cosine 0.9955, 28.67 dB, mean 3.90 for both builds.

Warm A/B, M5 Ultra, ReleaseFast `tf-h3-image-server` per build: main `82a2bfba3` against this card's SwiGLU-fused
head. Each session runs one resident server under `/tmp/tensorfold-model.lock` (`zsystem flock`, taken before the
server starts and held until it quits, 41-50 s a session) and sends the seed-42 pilot, then three unseen prompts (a
glassblower's workshop, a mountain railway station at dawn, a tide pool from above; seeds 21-23). Stage seconds:
prompt / transformer bind / 6 steps / decoder / total; the total also holds the case stage (0.01-0.07 s), so a row's
stages may sum short of it. Load is the one-minute average before / after each request:

| Run | main | load | this card | load |
|---|---|---|---|---|
| normal, pilot | 0.27 / 0.90 / 9.17 / 1.41 / 11.76 | 3.35 / 3.43 | 0.58 / 0.61 / 9.30 / 0.81 / 11.31 | 3.92 / 4.18 |
| normal, unseen 1 | 0.22 / 0.66 / 9.20 / 1.36 / 11.44 | 3.43 / 3.51 | 0.22 / 0.67 / 9.28 / 0.76 / 10.94 | 4.18 / 4.29 |
| normal, unseen 2 | 0.24 / 0.65 / 9.29 / 1.41 / 11.59 | 3.51 / 3.67 | 0.22 / 0.67 / 9.25 / 0.82 / 10.96 | 4.29 / 4.33 |
| normal, unseen 3 | 0.27 / 0.68 / 9.24 / 1.32 / 11.52 | 3.67 / 3.83 | 0.22 / 0.67 / 9.23 / 0.72 / 10.84 | 4.33 / 4.49 |
| fast (`--draft-steps 3`), pilot | 0.28 / 0.59 / 7.79 / 1.45 / 10.11 | 4.29 / 4.06 | 0.23 / 0.79 / 8.08 / 0.85 / 9.96 | 3.37 / 3.94 |
| fast, unseen 1 | 0.22 / 0.65 / 6.72 / 1.35 / 8.94 | 4.06 / 3.88 | 0.24 / 0.64 / 6.72 / 0.75 / 8.36 | 3.94 / 3.88 |
| fast, unseen 2 | 0.26 / 0.66 / 7.48 / 1.40 / 9.80 | 3.88 / 3.72 | 0.24 / 0.65 / 7.55 / 0.79 / 9.23 | 3.88 / 3.83 |
| fast, unseen 3 | 0.28 / 0.64 / 7.56 / 1.32 / 9.80 | 3.72 / 3.23 | spoiled twice, see below | |

The decoder stage falls from 1.32-1.45 s to 0.72-0.85 s, about 0.6 s a picture, and every clean warm total is lower.
The latents of the two builds are bit-identical request for request, so the steps' spread is the machine's.
Contention spoiled this card's fast unseen 3 twice. Both are kept in the receipt, not dropped. The first session
recorded decoder 1.23 s, total 9.77 s, with load rising from 3.83 to 6.11. A whole fast session rerun right after met
other workers' CPU tests: load rose from 5.71 to 21.14, and the session recorded decoder 1.37 / 0.84 / 0.93 / 1.04 s,
totals 11.03 / 9.83 / 9.95 / 9.93 s. Over the following hour the one-minute load stayed between 12 and 86.

The fast mode was rerun clean once the load fell below 7: main `4f805250e` against this card's head (same decoder
code), one locked fast session per build back to back (54 s and 39 s, lock held only for the session), the seed-42
pilot and three new unseen prompts (a blacksmith's forge at night, a lighthouse in a winter storm, a spice stall from
above; seeds 31-33). The exact text of the earlier unseen prompts was not kept, so all four requests ran again in
both builds. The session script, the prompts, both servers' logs, the per-request loads and the pictures' and
latents' SHA-256 are in [`docs/receipts/h3-image-decoder-152-fast`](../receipts/h3-image-decoder-152-fast/):

| Run | main | load | this card | load |
|---|---|---|---|---|
| fast, pilot | 0.28 / 0.86 / 7.81 / 1.35 / 10.32 | 6.47 / 6.15 | 0.28 / 0.58 / 7.70 / 0.75 / 9.31 | 5.15 / 4.81 |
| fast, unseen 1 | 0.22 / 0.65 / 7.56 / 1.37 / 9.86 | 6.15 / 5.58 | 0.22 / 0.63 / 6.75 / 0.79 / 8.40 | 4.81 / 4.54 |
| fast, unseen 2 | 0.22 / 0.65 / 7.50 / 1.43 / 9.81 | 5.58 / 5.34 | 0.22 / 0.66 / 7.62 / 0.82 / 9.33 | 4.54 / 4.07 |
| fast, unseen 3 | 0.22 / 0.63 / 7.57 / 1.32 / 9.75 | 5.34 / 4.98 | 0.22 / 0.66 / 7.59 / 0.72 / 9.20 | 4.07 / 3.98 |

The decoder stage drops by 0.54-0.64 s in each request, and every total comes in lower. The latents are
bit-identical request for request. The two builds' PNGs differ by the decoder's rounding alone: 53.3-56.6 dB, mean
0.14-0.29, max 5-9.

### Card 160: the Euler on the GPU, checked against the host's

I found this card's change already on main: card 137 (`e0d1072fb`) runs `h3_euler` after each step's head on the
resident `video_latents`, with one upload before step 1 and one download before the decoder (`runSteps` in
`tests/h3_image.zig`). What it lacked was a test: the kernel contract checks `h3_euler` against float64 within a
bound, and card 137's latent sha1 was a one-off receipt. I added `the steps' Euler on the GPU leaves main's
host-Euler latent bit for bit` to `test-h3-image-server` (in `zig build test`): on one binding of the synthetic
checkpoint it runs the six steps with the GPU Euler, then main's old loop (velocity down, `picture.euler` on the host,
the latent up each step) and requires the two latents' bytes equal. Scaling the host's delta by 0.999 turns it red.

I did not take a new warm 2048 A/B: this branch's render path is byte for byte main's, so it would time the same
binary twice. Card 137's A/B above is the measurement of this lever: flat on wall clock (head 11.13-11.74 s, base
11.06-11.79 s), since about 0.1 s a step of host round-trips does not show next to a 1.6 s step.

### Card 148: the Spark's latest four cards on the Metal path

Each Spark card checked against what this path already runs ([minimax-h3-cuda.md](minimax-h3-cuda.md) for the Spark
receipts). None leaves a Metal change to try that earlier cards have not already measured or that a profile does not
bound below run noise, so the render path is unchanged; the only code change is the counter-timed `profile` below:

| Spark card | Spark result | On the Metal path |
|---|---|---|
| 143, resident int8 text encoder | **Reverted**: prompt 1.22 → 0.16 s, but the fox missed card 80's rule (27.47 dB, mean 4.44) | Skipped: no win on the Spark. The prompt path stays bf16 (`hp_gemm`), resident, with kept states. |
| 144, decoder toward 0.7 s | Kept: `h3d_flash` fuses scores, softmax, p·v and unheads; 0.84 → 0.78 s GPU | Already here: `h3d_attention` has been one fused kernel since card 124 (point 2): fp32 scores, a max-subtracted softmax and p·v in one launch, two passes over the tile's keys rather than an online softmax. Card 152 took the decode to **0.49 s** of GPU, under the Spark's 0.7 s target. FP16 products lost here too (card 152). |
| 145, passes outside the products | Kept, then folded into 146: `h3_heads_i8` and `h3_swiglu_int8` read the int8 products' int32 sums 16 bytes at a time | Nothing to port: the int8 product here writes bf16 from its epilogue (card 136), so no int32 rows exist to dequantize. The bf16 passes left — `h3_heads` 0.10 s and `h3_swiglu` 0.10 s of the 9.20 s profiled below, their lanes' loads already coalesced — bound a wider-load rewrite to a few hundredths of a second, the size card 162 dropped as noise. `h3_norm_quant` (card 135) is the fusion that did carry. |
| 146, own int8 product | Kept: TMA ring, persistent grid in bands of eight, scales and SwiGLU in the epilogue; 171.4-189.4 TOPS, a step 2.08 → 1.77 s | Scales in the epilogue: here since card 136. SwiGLU in the epilogue: lost twice (card 135, 1.63-1.73 s against 1.53 s a step; card 150's spill receipts). TMA: no Metal counterpart; card 150's 4-byte operand loads won isolated and lost the served A/B. Banded walk: `swizzle` 6 already walks 64 of a 2048 picture's 65 row tiles under each weight tile (card 81); a persistent grid's other gain, overlapping a TMA prologue, needs the TMA this GPU lacks. |

Warm A/B, M5 Ultra, ReleaseFast `tf-h3-image-server` per build: main `d9d87fe67` against this card's head, whose
render path differs only in the profile switch (unprofiled, each kernel keeps its barrier; an encoder's last barrier
is new). One resident server a session, card 34's pilot prompt, seed 42, one cold then three warm 2048 requests,
each session holding `/tmp/tensorfold-model.lock` for its own 30-55 s, run back to back in one window with no other
model running; load is the one-minute average before → after. Every step of the five committed sessions (these four
and the profile below) is
`timing_valid=true` (a head session that speech interrupted, and a main fast session drawn while speech was playing,
were rerun once speech had been quiet for a minute).

| Session | Warm totals | 6 steps | Decoder | Load |
|---|---|---|---|---|
| main, normal | 9.97, 9.97, 9.96 s | 9.20 s | 0.75-0.76 s | 5.4 → 4.9 |
| head, normal | 9.96, 9.96, 9.96 s | 9.20-9.21 s | 0.75 s | 4.9 → 5.3 |
| main, fast (`--draft-steps 3`) | 6.82, 6.82, 6.83 s | 6.07-6.09 s | 0.74 s | 6.8 → 6.0 |
| head, fast (`--draft-steps 3`) | 6.82, 6.83, 6.83 s | 6.08 s | 0.74-0.75 s | 6.1 → 7.1 |

The head is as fast as main and draws the same bytes. Normal mode: every latent of both builds is sha1 `776b443a…`,
the value card 135 pinned, and every PNG `85098fa9…`; against episode 1's Mac latent and PNG
(`h3-image-runs/34/t2i-2048.latent.f32`, `fox-2048-seed42.png`, `904814d6…`) that pair gives cosine 0.995459, PSNR
28.6735 dB (card 150's 28.673 is the same value cut at three places, not rounded) and mean pixel diff 3.903, card 80's
rule. Fast mode draws a different picture on purpose: every latent
of both builds is `b4aece7d…` and every PNG `a2c9eb0c…`. Logs: `h3-image-fast-and-normal/mac-main-normal`,
`mac-normal`, `mac-main-drafts` and `mac-drafts` (`.server.log.txt`). These replace two earlier sessions at load
37-88, whose fast requests took 7.4-9.6 s on the same step GPU time.

In-picture profile: `tf-h3-image-server … profile` (`h3-image-fast-and-normal/mac-profile.server.log.txt`), retaken
after the fix that sums each kernel's command buffer into a step's GPU time, under the lock with no other model
running, load 4.7 → 4.5, one cold and three warm requests, every step `timing_valid=true`, latents and PNGs the normal
mode's bytes. Each kernel launch runs in its own encoder between two
GPU timestamp counter samples (`mtl.counters`, the stage boundary sampling this M5 supports), so a row is that
kernel's GPU time summed over its launches in the six steps and none of its neighbours'. A product, `h3_attention`
and the per-block passes launch 300 times (6 steps x 50 blocks); `h3_quantize_rows` 900, before qkv, attention out and
fc2 (fc1 takes `h3_norm_quant`'s rows). The last warm request's kernels add to 9.20 s; its six command-buffer-timed
steps to 9.18 s (1.53 s each), against the unprofiled 9.20 s.

The peaks are the silicon's, worked out for this M5 Ultra, as Apple publishes none: 80 GPU cores (`gpu-core-count`)
at the 1,620 MHz top GPU state (the IORegistry's `voltage-states9`), times 4,096 int8 ops and 2,048 bf16 flops a core
and cycle: **531 TOPS int8 and 265 TFLOPS bf16**. The per-cycle rates are card 150's chained-`mma_16x32` probe
(493 TOPS, 255 TFLOPS: 3,804 and 1,968 a core and cycle at that clock) rounded up to the powers of two.

| Kernel | Stage | Seconds (6 steps) | Rate | Of the silicon peak |
|---|---|---:|---:|---:|
| `h3_gemm_32x64_i8` | q k v gate | 2.064 | 138 TOPS | 26% |
| `h3_gemm_32x64_i8` | mlp in (fc1) | 2.729 | 140 TOPS | 26% |
| `h3_gemm_32x64_i8` | mlp out (fc2) | 1.573 | 121 TOPS | 23% |
| `h3_gemm_32x64_i8` | attention out | 0.777 | 123 TOPS | 23% |
| `h3_attention` | attention | 1.519 | 96 TFLOPS | 36% |
| `h3_quantize_rows` | quantize (900 launches: before qkv, out and fc2) | 0.153 | | |
| `h3_heads` | heads | 0.102 | | |
| `h3_swiglu` | swiglu | 0.098 | | |
| `h3_norm_quant` | norm quant | 0.082 | | |
| `h3_norm` | norm | 0.067 | | |
| `h3_head_norm` | head | 0.018 | | |
| `h3_rows_in` | rows in | 0.015 | | |
| `h3_gate_add` | gate add | 0.001 | | |
| `h3_euler` | euler | 0.000 | | |

The products run at card 150's isolated replay rates (138, 125, 139, 122 TOPS) inside the picture, so neighbouring
kernels do not slow them; at a quarter of the int8 peak the operand gather card 150 found is still what binds. The
memory passes below them are 0.54 s of the 9.20 s together.

### Card 170: the PNG written band by band while the GPU decodes

The resident server's decoder (`Vae.png` in `tests/h3_image_decoder.zig`, at the canvas's size) passes a `Bands` sink
(`image_decoder_host.zig`) to `Decoder.decode`. With a sink each pass is one tile row (11 tiles at 2048, 21 at 4096)
instead of up to 128. After a tile row is blended, every picture row above the next tile row's first is final.
`Bands.final` waits for the band before, then starts a CPU thread that turns the new rows to bytes
(`picture_vae.pixelRows`) and feeds them to `png.Stream`, the one zlib stream `png.encode` now runs on. The GPU decodes
the next tile row meanwhile; only the last band's deflate is left after the last pass. A resized request keeps the
whole-picture path.

**Gates.** `flate.Compress` drains only when its window fills, so the same rows in any split give the same bytes.
`png.zig`'s test streams uneven bands (one of zero rows) past the 64 KiB window against `encode`. `tf-h3-image-decoder
selftest` decodes 100 and 128 cells a side in passes of 128, of 16 and of a tile row with the bands streamed: bit-identical
pictures, and the banded PNG is `encode`'s bytes. ReleaseFast main (`403123903`) against this card, the standalone
decoder on retained latents: `cmp` finds the PNGs identical at 2048 (`92a08f22…`) and 4096 (`9498e225…`). There the
stage after the last pass falls from 0.23 to 0.07 s at 2048 and from 0.91 to 0.22 s at 4096, while the GPU time rises
from 0.54 to 0.56 s and from 1.75 to 1.82 s, since 11 or 21 smaller passes replace 1 or 4.

**Warm A/B, 22 Nov, M5 Ultra.** ReleaseFast `tf-h3-image-server` a side, one session a side under
`/tmp/tensorfold-model.lock` (`lockf`, held while the server runs, 52-97 s): card 34's pilot cold, then the harbour,
meadow and steam prompts warm and unseen, seed 42, 2048x2048. Each round is one session of each build, the first
alternating between them. Every PNG and latent is byte-identical between the sides. Decoder stage (GPU decode plus
PNG) and total, seconds, rounds 1 and 3:

| Request | main, round 1 | this card, round 1 | main, round 3 | this card, round 3 |
|---|---|---|---|---|
| cold pilot | 0.74 / 11.06 | 0.63 / 10.81 | 0.76 / 12.15 | 0.63 / 10.83 |
| warm harbour | 0.78 / 11.24 | 0.61 / 10.98 | 0.78 / 11.85 | 0.62 / 11.35 |
| warm meadow | 0.73 / 11.56 | 0.62 / 11.64 | 0.73 / 11.59 | 0.63 / 12.63 |
| warm steam | 0.76 / 10.66 | 0.62 / 10.57 | 0.76 / 10.97 | 0.66 / 11.48 |
| load (one minute) | 5.2-5.6 | 5.0-14.4 | 10.1-16.6 | 12.0-18.2 |

On a quiet machine the decoder stage falls from 0.73-0.78 s to 0.61-0.66 s on every request, 0.13 s on the mean: the
0.23-0.28 s after main's one pass becomes 0.06-0.11 s, less the 0.05 s the smaller passes add to the decode. The warm
totals also carry the prompt stage's CPU noise (0.67-1.65 s on either side, this card's meadow in round 3 losing 0.95 s
there). Other workers' builds loaded the machine through every other round. The warm totals of every completed round
(mean of the three warm requests, seconds, and the one-minute load across both sessions):

| Round | Load | main | this card |
|---|---|---|---|
| 1 | 5-14 | 11.15 | 11.06 |
| 2 | 16-34 | 14.14 | 11.30 |
| 3 | 10-18 | 11.47 | 11.82 |
| 6 | 19-35 | 11.99 | 11.48 |
| 8 | 23-38 | 16.30 | 11.37 |
| 9 | 34-50 | 16.50 | 19.57 |

The warm requests average 11.54 s on main and 11.46 s on this card over rounds 1, 3 and 6, the lowest one-minute load
at their warm requests; with round 2 (a hair lower over the whole session) in place of 6 they are 12.26 and 11.39 s,
and over all six 13.59 and 12.77 s. This card is faster in four rounds of six. Under load main's decoder stage
stretches to 0.87-3.82 s, its single-threaded deflate after the pass, while the band thread hides most of it behind the
GPU (round 9 loaded both). Rounds 4 (this card only), 5 and 7 (main only, load 57-99) lost their partner session to the
load and are left out.

### Card 174: fc2's rows split between the GPU and SME2, measured and stopped

The ideas page's idea 3: the CPU's SME2 units take a product's last rows while `h3_gemm_32x64_i8` runs the rest,
both on the same shared int8 operands and scales. The card's gate was a microbench first, kept only at 1.07x or more.
A throwaway bench (not kept) ran fc2, 5,376 x 14,336, at 4,120 and 16,384 rows: the shipped launch on the GPU's
rows, and on the CPU a pthread pool of SME2 32 x 32 tiles. Each tile ran four `smopa` int8 to int32 ZA tiles over
K-quad-packed operands, folded each 256-deep block at both scales in the GPU's order, then rounded to bf16. Times
are the wall from commit to both sides done, warm medians of eight, under `/tmp/tensorfold-model.lock`, 22 Nov, M5
Ultra, load 26-33. The CPU rows match the GPU's bf16 output bit for bit, so the picture gate cannot fail; speed is
what stops it:

| Rows | CPU rows (12 threads) | GPU alone | Split wall | Speedup |
|---|---|---:|---:|---:|
| 4,120 | 32 | 5.43 ms | 5.36 ms | 1.013x |
| 4,120 | 64 | 5.43 ms | 5.35 ms | 1.015x |
| 4,120 | 96 | 5.43 ms | 5.41 ms | 1.004x |
| 4,120 | 160 | 5.43 ms | 7.95 ms | 0.68x |
| 16,384 | 160 | 18.20 ms | 18.18 ms | 1.001x |
| 16,384 | 320 | 18.20 ms | 18.06 ms | 1.008x |
| 16,384 | 480 | 18.20 ms | 21.76 ms | 0.84x |

The GPU runs fc2 at 123 TOPS at 4,120 rows and 140 at 16,384. To reach 1.07x the CPU would need 8.6 and 9.8 TOPS
beside it. These CPU rates were measured:

- `smopa` alone, four independent ZA tiles: 3.1 TOPS on one thread and 10.4 on twelve (load ~14).
- The real tile, with operand loads and the fold, ran alone at 0.68 TOPS on one thread, 3.0-3.6 on twelve and
  5.4-5.8 on 24-36.
- Beside the GPU, twelve threads ran at 2.2-3.4 TOPS. The E-cores' threads (36) collapsed to 0.5-1.7 TOPS and
  slowed the GPU's own rows.

Even the bare `smopa` peak sits at the bar, before any load, fold or contention. The other three products run the
same kernel and tile on the GPU at 123-139 TOPS (card 150), so their bar is also 8.6 TOPS or more, about three times
the best real tile beside the GPU: no product split reaches 1.07x. The resident server change and its
warm A/B were not built. One build note for any later SME2 work here: Apple clang's `-march=armv9-a` emits
non-streaming SVE (`cntd`) into the streaming prologue, and the M5 traps it (SIGILL); `-march=armv8.6-a+sme2` builds
and runs.

### Card 180: step caching, measured and reverted

A `--step-cache X` option on `tf-h3-image-server` skipped a step's transformer pass and stepped the resident latent
on the last run step's velocity (x += delta v on the host), with a gate on the latent's change deciding when.
`--step-cache 0` ran every step and drew main's bytes (every latent and PNG `cmp`-identical on all four prompts), and a
synthetic-checkpoint test held the cached run bit for bit to a host loop at every gate between the measured moves. No
gate kept card 80's picture rule, so the code is reverted.

Two gates were tried, each against main `e0c33d47b` on card 34's pilot cold, then the harbour, meadow and steam
prompts warm and unseen (seed 42, 2048x2048). Each build ran as one resident ReleaseFast server, holding
`/tmp/tensorfold-model.lock` for its own 66-143 s session:

- **The latent's own move since the last run step** (relative L2). Step by step the latent moves 0.044-0.046, then
  0.11, 0.20, 0.35 and 0.51-0.53, the same on all four prompts. A gate of 0.08 reuses step 1's velocity for step 2
  only, and that one skip broke every picture: latent cosine 0.902-0.967, PSNR 15.9-20.6 dB, mean pixel difference
  15.8-23.0. Each step has its own distilled head, so step 1's velocity is not step 2's.
- **The clean picture the last two run steps predict** (x + sigma v, relative L2), reusing the last velocity for every
  later step once it settles. It never settles: it still moves 0.39-0.48 at step 3, 0.24-0.28 at 4, 0.16-0.20 at 5 and
  0.15-0.19 at 6. A gate of 0.2 skips steps 5 and 6 (pilot, harbour, steam) or step 6 alone (meadow). The meadow
  picture comes closest, at cosine 0.9946 but PSNR 21.76 dB and mean 13.7. The others reach 0.974-0.977, 19.9-23.8 dB
  and 10.3-14.9.

Every warm request, seconds (prompt / 6 steps / decoder / total; `F` marks a request with a step reporting
`timing_valid=false`). Main's session ran with no speech. Speech interrupted most of the branch's, and my own
`test-h3-image-server` run, outside the lock, overlapped the 0.2 session's harbour and meadow:

| Request | main | `--step-cache 0` | `--step-cache 0.2` | Load, main / 0 / 0.2 |
|---|---|---|---|---|
| cold pilot | 0.38 / 10.29 / 0.70 / 14.50 | 0.65 / 10.41 / 0.68 / 15.01 F | 0.45 / 9.84 / 13.81 / 27.45 F | 109 / 114 / 111 |
| harbour | 0.22 / 9.00 / 1.55 / 11.86 | 0.25 / 10.41 / 0.68 / 13.02 F | 5.00 / 19.36 / 7.21 / 36.19 F | 147 / 114 / 82 |
| meadow | 0.22 / 9.30 / 0.68 / 11.28 | 0.38 / 10.05 / 0.68 / 12.28 F | 10.75 / 34.66 / 6.51 / 55.68 F | 138 / 111 / 60 |
| steam | 0.22 / 9.29 / 0.67 / 10.19 | 0.22 / 9.93 / 0.81 / 10.98 | 4.18 / 6.68 / 1.05 / 11.93 | 134 / 113 / 96 |

Steam is the one clean pair: its two skipped steps took the six steps from 9.93 to 6.68 s, about 1.6 s a skip, while
the prompt's 4.18 s at load 96 cost the total more than that. With the speed in reach but no skip keeping the
picture, the bar is closed by quality, not time.

A first round of the latent-move gate at 0.08 measured warm totals of 23.65, 10.92 and 8.66 s, with six steps of 13.60,
8.50 and 7.72 s against `--step-cache 0`'s 11.89, 12.09 and 10.98 s, under the same quality failure. Its logs were
overwritten by the second round, so its `timing_valid` lines were never checked and these times are not a receipt.

#### Second attempt: the first block's residual (FBCache)

The untried form above: run block 0 every step and compare its MLP branch, relative L1 over the picture rows, with the
last fully run step's. Under the gate, skip blocks 1-49 and add that step's residual (the stream after the last gate
add minus the stream after block 0) to this step's stream, then run this step's own final norm and distilled head.
The per-step AdaLN of block 0 and the head stay live, so only the 49 blocks' work is reused. Same main `e0c33d47b`,
prompts and lock as above; one session a build, 69-104 s each, every request listed (receipts in
`h3-image-runs/180`, outside the repo):

- **`--step-cache 0`** ran every block and drew main's bytes: all four latents identical to main's, PSNR infinite.
- **The probe never settles.** Block 0's branch moves 0.49-0.52 at step 2, 0.49 at 3, 0.55-0.57 at 4, 0.68-0.72 at 5
  and 0.75-0.80 at 6, the same on all four prompts. FBCache's published gates are 0.08-0.12 on 20-50-step models;
  here every step moves four to ten times that.
- **`--step-cache 0.5`** skips step 3 alone, the quietest, on every prompt. That one skip breaks every picture:

| Prompt | latent cosine | PSNR | mean pixel diff |
|---|---|---|---|
| pilot | 0.983510 | 23.510 dB | 9.489 |
| harbour | 0.973617 | 19.755 dB | 11.760 |
| meadow | 0.991200 | 24.489 dB | 9.434 |
| steam | 0.988035 | 25.447 dB | 7.268 |

Better than reusing a velocity (15.9-20.6 dB for one skip), still 2.8-8.4 dB under card 80's 28.2 floor. Any
gate that skips more steps or later ones reuses an older residual against a larger move, so no gate in reach passes.

Seconds (prompt / 6 steps / decoder / total; `F` marks a request with a step reporting `timing_valid=false`):

| Request | main | `--step-cache 0` | `--step-cache 0.5` |
|---|---|---|---|
| cold pilot | 0.39 / 10.31 / 0.68 / 12.74 | 0.29 / 9.89 / 1.37 / 12.44 F | 1.09 / 14.29 / 2.61 / 19.97 F |
| harbour | 0.22 / 9.78 / 0.66 / 12.18 | 0.32 / 13.97 / 1.95 / 17.53 | 0.80 / 21.08 / 2.62 / 27.03 F |
| meadow | 0.22 / 9.27 / 0.67 / 11.13 | 0.53 / 15.78 / 2.91 / 21.72 | 2.12 / 17.50 / 5.30 / 29.90 F |
| steam | 0.22 / 9.27 / 0.68 / 10.18 | 0.91 / 16.49 / 2.91 / 20.40 | 0.34 / 8.93 / 0.69 / 9.98 |

Load was 63-71 through main's session and 35-71 through the branch's, with other builds on the CPU. Steam's steps under
0.5 are the one all-valid pair: 1.68, 1.66, **0.14** (skipped), 1.71, 1.94 and 1.78 s against main's 1.54 s each,
8.93 s against 9.27 s. A skip saves its 49 blocks, but the host's pass over the stream (block 0 drained, its 22M bf16
values widened, compared and kept each step) costs 0.1-0.4 s on every run step, so one skip nets about 0.3 s, and
moving that pass to the GPU would buy at most 1.5 s for a picture that fails. Its 9.98 s total, the best of either
design and under main's 10.18 s here, is one request with a failed picture, so it is claimed as nothing. The code is
reverted.

### Next, ranked

| # | Idea | Size | Basis |
|---|---|---|---|
| 1 | SGLang's conditioning cache | under -2 s at warm 2048 | Residency landed with card 50: warm 2048 went from 27 s to 16.4-16.6 s, more than the 6.3 s of loader logs the projection counted, since each one-shot tool also re-reads its files and rebuilds its kernels. A conditioning cache would skip only the prompt (1.2-2.0 s warm); it needs separate accounting and cannot omit prompt work in an uncached comparison. |
| 2 | PNG writing | landed overlapped (card 170, -0.13 s at 2048); upper estimate -18 s at 4096, -4 s at 2048 in parallel bands | Card 170 streams each tile row's band into one `flate.Compress` while the GPU decodes the next: the bytes stay main's. Parallel bands could remove much more but change compressed bytes, so they do not satisfy the PNG byte-identity gate; a decoded-pixel gate in its place needs explicit approval before runs. |
| 3 | int8 weights and activations on the M5's tensor units | about -30 s of the 60 s of steps at 4096 | The steps are compute-bound: a 2048 step is about 183 TFLOP in 1.83 s, a 4096 step about 1,020 TFLOP in 10 s, both near 100 TFLOPS of bf16. int8 `matmul2d` at twice bf16's rate would halve the GEMMs. ComfyUI's int8 here shows the other half: dequantized before each product, its steps are 1.7-1.9x slower than our bf16, so int8 weights alone buy only bytes. Not bit-identical; it needs its own latent gate. SGLang's FP8 on an RTX PRO 6000: -32% at 2048, -22% at 4K. |
| 4 | Attention | up to -15 s at 4096 | At 16,408 rows attention is about 38% of a step's work, growing with the square of the rows: 4096's step is 5.4x 2048's for 4x the rows. SGLang's fused QK-norm and rotary saves under 1% of a step here. |
| 5 | Step caching (TeaCache, FBCache, cache-dit) | measured, lost twice (card 180) | Neither form keeps card 80's rule. Reusing the last velocity: one skip at step 2 gives 15.9-20.6 dB, step 6 alone 21.76 dB. Reusing blocks 1-49's residual behind a first-block gate (FBCache): block 0's branch moves 49-80% a step, four to ten times FBCache's gates, and skipping step 3 alone gives 19.8-25.4 dB against the 28.2 floor. Six distilled steps leave no step close enough to its neighbour to reuse; closed. |
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
