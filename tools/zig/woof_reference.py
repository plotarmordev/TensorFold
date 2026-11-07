"""Woof's numpy runtime on the parity texts, for zig-out/bin/tf-woof-parity. Writes OUT/<n>.f32 (raw float32 mono 24 kHz)
for each line of TEXTS, seeded <seed>, through the same pieces and redraw check the engine runs; the normalizer is off,
as the engine speaks text as given. Usage: woof_reference.py WOOF_DIR TEXTS OUT [--seed N]"""
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
