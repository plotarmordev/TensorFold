# Underdog Woof

This page describes the native engine. `woof` serves [Underdog Woof](https://huggingface.co/ConwayResearch/Underdog-Woof),
a small English text-to-speech model with two voices (Juno and Mira), on the CPU of an Apple-silicon Mac or a Linux arm64 machine at 24 kHz. It is a port of
Woof's numpy runtime, not a build on Drama 3: it shares only the server's `/v1/audio/speech` route.
The family's upstream proposal and gate are [ashhart/TensorFold#2504](https://github.com/ashhart/TensorFold/issues/2504).

Underdog Woof is by Underdog ([underdog.ai](https://underdog.ai)), under the
[Underdog Non-Commercial License 1.0](https://huggingface.co/ConwayResearch/Underdog-Woof/blob/main/LICENSE): free for
non-commercial use with credit, commercial use needs a licence from license@conway.tech. Audio you publish from it
must name "Underdog Woof" or "underdog.ai" where it is posted, and an app that plays it must show "Voice by Underdog
Woof" with a link to underdog.ai. TensorFold ships no Woof weights; the checkpoint keeps its own licence.

## Checkpoint

The engine reads `woof_int8.npz` (40.6 MB) from the model folder, as the Hugging Face download ships it; a folder with
that file and no `config.json` is the `woof` family. Tested revision: `ConwayResearch/Underdog-Woof` at
`be702960539777fc3f698e54335276a8391b00f9`; sha256 of `woof_int8.npz`
`d895fbbe84b8f4f461e3e94ab48b1164353b1988ce8d365f5aa8263d0de1203e`, of the numpy runtime `underdog_woof/runtime.py`
`e22d1e2a92f4d28c3784abbb889d77609dcecd1856bf6e65f03432b803a8d220`.

```bash
hf download ConwayResearch/Underdog-Woof --revision be702960539777fc3f698e54335276a8391b00f9 --local-dir WOOF_DIR
```

- Speech model: 8 transformer layers 448 wide with the voice's 16-row prefix, then a 4-block one-step flow head that
  turns noise into a 32-wide latent frame (25 a second), with the end-of-speech probability beside it.
- Decoder: a 2-layer transformer and a SEANet stack that upsample each latent frame to 960 samples.
- Weights: the speech linears int8 with a scale per row; the decoder's int8 in groups of 32. Both are dequantized once
  at load and packed into 16-output panels for the fp32 matvecs.

## Serve

```bash
zig build native
zig-out/native/bin/tensorfold-native serve /path/to/underdog-woof --port 8399
curl -s localhost:8399/v1/voices
curl -s localhost:8399/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "underdog-woof", "input": "Your order ships on March 3rd.", "voice": "mira"}' -o out.wav
```

`/v1/voices` lists `default` (Juno), `juno` and `mira`. `--voices DIR` adds each `DIR/NAME.npy`, a float32 (16, 448)
voice prefix as Woof saves one, after the built-ins; `--default-voice NAME` picks the default. `response_format` is
`wav` or `pcm`, and `stream: true` with `pcm` sends each sentence as soon as it is spoken. Text, vision, cache and device
flags are refused at startup.

Like Woof's own `Woof()` call, the engine speaks text as given:

- **Pieces.** Each sentence is spoken on its own; a sentence over 120 characters is cut at clause punctuation and merged
  greedily, then at a conjunction near the middle (`textsplit.py`). A piece still over 315 characters, whose speech at
  the slowest accepted pace would outrun its 30 s, is cut at its last space within 315 (where numpy fails or stops
  early).
- **Redraw on stall.** A piece's take is redrawn, up to 3 times, when it holds a quiet run of 1 s or more between voiced
  40 ms frames, runs past 0.095 s a character, or hits the 30 s piece cap; the first clean take wins, else the best by
  (faults, longest quiet run, pace). A reply whose kept take for any piece hit that cap, so its speech was cut short,
  finishes with `length` rather than `stop`.
  `TF_WOOF_DIR=WOOF_DIR zig build test` also runs that check on the checkpoint (about seven minutes, in Debug).
- **Seeds.** The noise is torch's `manual_seed` + `randn` stream, so a seed draws what Woof's runtime draws. The server
  seeds each request from its input, as it does for Drama, or from the request's `seed`. Woof's flow temperature is
  fixed at 0.5, so `temperature`, `top_p` and `top_k` are refused.

Woof's text normalizer (numbers, dates and prices to words) is not ported: send spoken-form text.

## Parity with the numpy runtime

```bash
python tools/zig/woof_reference.py WOOF_DIR zig/tests/fixtures/woof/parity.txt /tmp/woofref --seed 7
zig build tf-woof-parity && zig-out/bin/tf-woof-parity WOOF_DIR zig/tests/fixtures/woof/parity.txt /tmp/woofref 7
```

The check: each text's first second of audio within 2e-3 of numpy's samples, and its length within 4 frames (160 ms).
On the three parity texts (seed 7) the lengths are equal and the first second agrees to 5.8e-5, 6.8e-5 and 2.3e-4; the
two short texts agree to 2.7e-4 over their whole length. Past the first second no fixed seed gives equal samples on long
pieces: the speech loop feeds each frame back, and moving numpy's own weights by 1e-7 changes a later latent by 1.1 and a
piece's length by a frame. The long text's third clause shows the same drift between the engine and numpy.

## Mac figures

**Headline: 1.4x faster than running Woof the normal way on the same Mac (Juno; 1.25x for Mira), with the audio the
one-thread engine makes.** Threads change no sample; against numpy the audio drifts as the parity section says, with
equal word errors. It rests on one Apple M5 Ultra (256 GB, 12 performance cores), the 24-text English bench
(`zig/tests/fixtures/woof/bench-en.jsonl`, Drama's `drama-bench-en-v1`: 8 short, 8 medium, 8 long), seed 0, one
take each, Juno and Mira, run back to back on an idle machine. "The normal way" is Woof's own `Woof()` call from its
repo (numpy 2.5 on Apple's Accelerate, default threads, normalizer on; it changes none of the 24 texts). The lead comes
from TensorFold using 12 cores where numpy keeps one busy: on one thread TensorFold is 2.5x slower than numpy (0.151 to
0.060 Juno). Underdog ships no torch path for the two-voice model, so there is nothing else to race.

| | TensorFold `woof`, Juno | Woof numpy, Juno | TensorFold `woof`, Mira | Woof numpy, Mira |
| --- | --- | --- | --- | --- |
| Real-time factor | **0.0426** | 0.0595 | **0.0547** | 0.0683 |
| First audio, p50 | **0.147 s** | 0.191 s | **0.199 s** | 0.327 s |
| First audio, p90 | **0.263 s** | 0.353 s | **0.499 s** | 0.792 s |
| Peak memory (RSS) | **255 MiB** | 384 MiB | **255 MiB** | 372 MiB |
| Threads (CPU seconds per wall second) | 12 (10.5) | Accelerate default (0.99) | 12 (10.6) | Accelerate default (1.01) |
| Word errors (Whisper large-v3, 670 words) | 7 (1.04%) | 7 (1.04%) | 6 (0.90%) | 6 (0.90%) |
| Speaker similarity to the voice's samples (ECAPA) | 0.798 | 0.798 | 0.817 | 0.819 |

With `VECLIB_MAXIMUM_THREADS=1` numpy gives 0.0628 and 0.209 s for Juno: Accelerate's threads do not help these
one-frame matvecs. TensorFold's samples are the same bits at 1 and 12 threads.

```bash
zig build tf-woof-bench && zig-out/bin/tf-woof-bench WOOF_DIR zig/tests/fixtures/woof/bench-en.jsonl runs/tensorfold 0 juno
python tools/zig/woof_numpy_bench.py WOOF_DIR zig/tests/fixtures/woof/bench-en.jsonl runs/numpy --voice juno
python tools/zig/woof_score.py scores.json runs/tensorfold runs/numpy --corpus zig/tests/fixtures/woof/bench-en.jsonl \
  --whisper WHISPER_LARGE_V3_DIR --ecapa SPKREC_ECAPA_VOXCELEB_DIR \
  --ref WOOF_DIR/samples/juno_0.wav --ref WOOF_DIR/samples/juno_1.wav --ref WOOF_DIR/samples/juno_2.wav
```

Both benches write each take to `OUT/<id>.wav` and print first audio (the first piece spoken) p50 and p90, the
real-time factor, peak RSS and CPU seconds per wall second; `tf-woof-bench` takes `[SEED] [VOICE] [THREADS]`, 0 threads
meaning the performance cores, as the server runs. `woof_score.py` gives the word error rate (Whisper large-v3) and
speaker similarity (ECAPA against the voice's three samples), the scoring Woof's own bench uses.

Where the time goes, on one 145-frame piece (ms per 40 ms frame):

| | Speech model | Decoder |
| --- | --- | --- |
| Woof numpy | 1.1 | 1.1 |
| TensorFold, 1 thread | 2.5 | 2.6 |
| TensorFold, 12 threads | 0.7 to 1.0 | 0.47 |

On one thread Accelerate's matvecs beat the engine's NEON panels 2.3x per frame. The engine splits each matvec over a
persistent pool by output panels (by 4-row blocks for the prompt and decoder), the speech attention by head and the
decoder's ELU and GELU by runs, so every output sums in the same order as on one thread. Past 6 threads the speech model
stops scaling; each of its frames is about 50 small jobs.

## Linux figures

The CUDA binary links the same CPU engine, and it runs with no GPU. On a DGX Spark (GB10, Linux arm64) in a container
held to 8 of its 20 cores and 7 GiB, with no nvcc (so no CUDA kernels): `zig build native` and `tf-woof-bench` build,
`TF_WOOF_DIR=… zig build test` passes Woof's 13 tests on the checkpoint, and `serve` speaks WAV and streamed PCM, the
stream equal to the whole reply. Same 24 texts, seed 0, one take each:

| | TensorFold `woof`, Juno | Woof numpy, Juno | TensorFold `woof`, Mira | Woof numpy, Mira |
| --- | --- | --- | --- | --- |
| Real-time factor, 1 thread | 0.157 | 0.258 | 0.209 | 0.290 |
| First audio p50, 1 thread | 0.511 s | 0.812 s | 0.928 s | 1.377 s |
| Real-time factor, 8 threads | **0.0509** | | **0.0629** | |
| First audio p50, 8 threads | **0.167 s** | | **0.276 s** | |
| Peak memory (RSS) | **203 MiB** | 270 MiB | **202 MiB** | 278 MiB |

Numpy is 2.5.4 on OpenBLAS with `OPENBLAS_NUM_THREADS=1`; at its default it starts a thread per core and ran so slowly
under the quota that the run was stopped. The engine's default, 0 threads, which `serve` always uses, has the same
fault: it counts the 20 cores the affinity mask shows rather than the 8 the CPU quota allows, and gives 0.94 (Juno)
and 1.28 (Mira). `tf-woof-bench` takes a thread count; the server has none yet.
