"""Woof's numpy runtime over a speech bench corpus, timed as tf-woof-bench times the engine: OUT/<id>.wav, first audio
(the first piece spoken), real-time factor, peak RSS and CPU seconds per wall second. VECLIB_MAXIMUM_THREADS caps its
Accelerate threads."""
import argparse, json, resource, statistics, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("woof", type=Path, help="Underdog Woof checkout (underdog_woof/ and woof_int8.npz)")
ap.add_argument("corpus", type=Path, help="JSON lines with id and text")
ap.add_argument("out", type=Path)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--voice", default="juno")
args = ap.parse_args()
sys.path.insert(0, str(args.woof))
from underdog_woof import SR, Woof

tts = Woof(voice=args.voice)
piece, first = tts._piece, []


def timed_piece(text):
    wav = piece(text)
    if len(first) < item_count:
        first.append(time.perf_counter() - start)
    return wav


tts._piece = timed_piece
args.out.mkdir(parents=True, exist_ok=True)
wall = audio = 0.0
cpu0 = time.process_time()
for item_count, line in enumerate(args.corpus.read_text().splitlines(), 1):
    item = json.loads(line)
    start = time.perf_counter()
    wav = tts(item["text"], seed=args.seed)
    took = time.perf_counter() - start
    wall, audio = wall + took, audio + len(wav) / SR
    tts.save(str(args.out / f"{item['id']}.wav"), wav)
    print(f"{item['id']}: first audio {first[-1]:.3f} s, total {took:.3f} s, audio {len(wav) / SR:.2f} s", flush=True)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
q = statistics.quantiles(first, n=10, method="inclusive")
p50, p90 = q[4], q[8]
print(f"first audio p50 {p50:.3f} s p90 {p90:.3f} s, real-time factor {wall / audio:.4f}, peak RSS {rss:.1f} MiB, "
      f"cores busy {(time.process_time() - cpu0) / wall:.2f}")
