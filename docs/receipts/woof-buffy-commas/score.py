"""One row a take, as tools/zig/woof_score.py scores them: Whisper large-v3 against the corpus text through Whisper's
English normalizer, ECAPA cosine to the voice's samples. A text spoken as units (ID.1, ID.2...) is scored on its
units' audio joined in order.

    python score.py OUT.jsonl CORPUS WHISPER_DIR ECAPA_DIR SAMPLES_DIR RUN_DIR...    # RUN_DIR named NAME-VOICE-SEED
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from transformers import pipeline

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for name in ("out", "corpus", "whisper", "ecapa", "samples"):
    ap.add_argument(name, type=Path)
ap.add_argument("runs", nargs="+", type=Path)
args = ap.parse_args()

asr = pipeline("automatic-speech-recognition", model=str(args.whisper), device="cpu", dtype=torch.float32)
ecapa = EncoderClassifier.from_hparams(source=str(args.ecapa), overrides={"pretrained_path": str(args.ecapa)},
                                       local_strategy=LocalStrategy.NO_LINK, run_opts={"device": "cpu"})


def errors(ref, hyp):
    """Substitutions, deletions and insertions of a minimal word alignment, fewest edits first."""
    row = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, r in enumerate(ref, 1):
        prev, row = row, [(i, 0, i, 0)]
        for j, h in enumerate(hyp, 1):
            same = prev[j - 1]
            row.append(min((same[0] + (r != h), same[1] + (r != h), same[2], same[3]),
                           (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3]),
                           (row[j - 1][0] + 1, row[j - 1][1], row[j - 1][2], row[j - 1][3] + 1)))
    return dict(zip(("sub", "del", "ins"), row[-1][1:]))


def embed(y16):
    with torch.inference_mode():
        e = ecapa.encode_batch(y16[None])[0, 0].numpy()
    return e / np.linalg.norm(e)


def load16k(paths):
    reads = [soundfile.read(p, dtype="float32") for p in paths]
    return AF.resample(torch.from_numpy(np.concatenate([y for y, _ in reads])), reads[0][1], 16000)


texts = {json.loads(line)["id"]: json.loads(line)["text"] for line in args.corpus.read_text().splitlines()}
rows = []
for run in args.runs:
    voice, seed = run.name.split("-")[-2:]
    ref = np.mean([embed(load16k([p])) for p in sorted(args.samples.glob(f"{voice}_*.wav"))], 0)
    for item, text in texts.items():
        units = sorted(run.glob(f"{item}.*.wav"), key=lambda p: int(p.stem.split(".")[-1])) or [run / f"{item}.wav"]
        y16 = load16k(units)
        hyp = asr({"raw": y16.numpy(), "sampling_rate": 16000}, chunk_length_s=30,
                  generate_kwargs={"language": "english", "task": "transcribe"})["text"].strip()
        words = asr.tokenizer.normalize(text).split()
        rows.append({"run": run.name, "voice": voice, "seed": int(seed), "item": item, "words": len(words),
                     **errors(words, asr.tokenizer.normalize(hyp).split()),
                     "cos": round(float(embed(y16) @ ref / np.linalg.norm(ref)), 4),
                     "seconds_per_char": round(len(y16) / 16000 / len(text), 4), "hypothesis": hyp})
    print(run.name, flush=True)
args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
