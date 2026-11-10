"""Word error rate and speaker similarity of take folders: Whisper edit distance and ECAPA cosine."""
import argparse, json
from pathlib import Path
import numpy as np
import soundfile
import torch
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from transformers import pipeline

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("runs", nargs="+", type=Path)
ap.add_argument("--corpus", type=Path, required=True)
ap.add_argument("--whisper", type=Path, required=True)
ap.add_argument("--ecapa", type=Path, required=True)
ap.add_argument("--ref", type=Path, action="append", required=True)
args = ap.parse_args()


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def load16k(path):
    y, rate = soundfile.read(path, dtype="float32")
    return AF.resample(torch.from_numpy(y), rate, 16000)


def embed(y16):
    with torch.inference_mode():
        e = ecapa.encode_batch(y16[None])[0, 0].numpy()
    return e / np.linalg.norm(e)


asr = pipeline("automatic-speech-recognition", model=str(args.whisper), device="cpu", torch_dtype=torch.float32)
ecapa = EncoderClassifier.from_hparams(source=str(args.ecapa), overrides={"pretrained_path": str(args.ecapa)},
                                       local_strategy=LocalStrategy.NO_LINK, run_opts={"device": "cpu"})
texts = {json.loads(l)["id"]: json.loads(l)["text"] for l in args.corpus.read_text().splitlines()}
ref = np.mean([embed(load16k(p)) for p in args.ref], 0)
ref /= np.linalg.norm(ref)
result = {}
for run in args.runs:
    errors = words = 0
    cosines = []
    for item, text in texts.items():
        y16 = load16k(run / f"{item}.wav")
        hyp = asr({"raw": y16.numpy(), "sampling_rate": 16000}, chunk_length_s=30,
                  generate_kwargs={"language": "english", "task": "transcribe"})["text"]
        r, h = asr.tokenizer.normalize(text).split(), asr.tokenizer.normalize(hyp).split()
        errors, words = errors + edit_distance(r, h), words + len(r)
        cosines.append(float(embed(y16) @ ref))
    result[run.name] = {"wer": round(errors / words, 5), "errors": errors, "words": words,
                        "speaker_cos_mean": round(float(np.mean(cosines)), 4)}
    print(run.name, json.dumps(result[run.name]), flush=True)
args.out.write_text(json.dumps(result, indent=1))
