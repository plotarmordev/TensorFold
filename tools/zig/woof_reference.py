"""Woof's numpy runtime for tf-woof-parity: writes OUT/<n>.f32 per TEXTS line, seeded, normalizer off."""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("woof", type=Path, help="Underdog Woof checkout (underdog_woof/ and woof_int8.npz)")
ap.add_argument("texts", type=Path, help="one text a line")
ap.add_argument("out", type=Path)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
sys.path.insert(0, str(args.woof))
from underdog_woof import Woof

tts = Woof(normalize=False)
args.out.mkdir(parents=True, exist_ok=True)
for n, text in enumerate(args.texts.read_text().splitlines()):
    tts(text, seed=args.seed).tofile(args.out / f"{n}.f32")
