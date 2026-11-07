# Underdog Woof

This page describes the native engine. `woof` serves [Underdog Woof](https://huggingface.co/ConwayResearch/Underdog-Woof),
a small English text-to-speech model with two voices (Juno and Mira), on the Mac's CPU at 24 kHz. It is a port of
Woof's numpy runtime, not a build on Drama 3: it shares only the server's `/v1/audio/speech` route.

Underdog Woof is by Underdog ([underdog.ai](https://underdog.ai)), under the
[Underdog Non-Commercial License 1.0](https://huggingface.co/ConwayResearch/Underdog-Woof/blob/main/LICENSE): free for
non-commercial use with credit, commercial use needs a licence from license@conway.tech. Audio you publish from it
must name "Underdog Woof" or "underdog.ai" where it is posted, and an app that plays it must show "Voice by Underdog
Woof" with a link to underdog.ai. TensorFold ships no Woof weights; the checkpoint keeps its own licence.

## Checkpoint

The engine reads `woof_int8.npz` (40.6 MB) from the model folder, as the Hugging Face download ships it; a folder with
that file and no `config.json` is the `woof` family.

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
  greedily, then at a conjunction near the middle (`textsplit.py`).
- **Redraw on stall.** A piece's take is redrawn, up to 3 times, when it holds a quiet run of 1 s or more between voiced
  40 ms frames, runs past 0.095 s a character, or hits the 30 s piece cap; the first clean take wins, else the best by
  (faults, longest quiet run, pace).
- **Seeds.** The noise is torch's `manual_seed` + `randn` stream, so a seed draws what Woof's runtime draws. The server
  seeds each request from its input, as it does for Drama.

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

An Apple M5 Ultra (256 GB), the 24-text English bench (`zig/tests/fixtures/woof/bench-en.jsonl`, Drama's
`drama-bench-en-v1`: 8 short, 8 medium, 8 long; Juno, seed 0, one take each), against Woof's numpy runtime on one
thread as its own bench runs it (its normalizer changes none of the 24 texts):

| | TensorFold `woof` | Woof numpy |
| --- | --- | --- |
| First audio, median | 0.53 s | 0.23 s |
| Real-time factor | 0.162 | 0.066 |
| Peak memory (RSS) | 252 MiB | 400 MiB |
| Word error rate (Whisper large-v3, 670 words) | 1.04% (7) | 1.04% (7) |
| Speaker similarity to Juno's samples (ECAPA cosine) | 0.798 | 0.798 |

```bash
zig build tf-woof-bench && zig-out/bin/tf-woof-bench WOOF_DIR zig/tests/fixtures/woof/bench-en.jsonl OUT 0
```

`tf-woof-bench` writes each take to `OUT/<id>.wav` for scoring and prints first audio (the first piece spoken), the
real-time factor and peak RSS. Word error rate and speaker similarity come from Whisper large-v3 and ECAPA over those
files, the scoring Woof's own bench uses. The engine is single-threaded fp32 with its own packed matvecs; this numpy
links Apple's Accelerate, whose matmuls make it 2.4x faster per second of audio.
