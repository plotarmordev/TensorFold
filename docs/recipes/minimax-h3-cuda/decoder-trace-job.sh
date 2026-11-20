#!/bin/sh
# Card 144: the --tc decoder's warm 2048² decode kernel by kernel beside mmh3's VAE, in one TOP
# window — bench-trace-job.sh's our-side trace, on main's decoder (bench-144-base.patch reversed) and
# on this commit's, plus mmh3-server-job.sh's, so both engines' decoder kernel budgets sit in one log.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/decoder-trace-job.sh
# Our side: bench-script.sh under the CUPTI preload; the decoder server's cupti-trace.csv.<pid> holds
# every kernel of the warm decodes, start and end ns, so the tables show gaps as well as kernel times.
# mmh3's side: mmh3-server-job.sh unchanged; its warm picture's trailing stretch after the last
# >20 ms idle gap is the VAE.
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-144-base.patch
cat >/work/decoder_kernels.py <<'PY'
# Print the last warm decode's kernel table from a CUPTI trace: split at >400 ms idle gaps (the
# pictures), keep the last segment, then (with --tail) only its stretch after the last >20 ms gap.
import csv, sys
rows = sorted((int(r["start_ns"]), int(r["end_ns"]), r["name"], "%sx%sx%s" % (r["gridX"], r["gridY"], r["gridZ"]))
              for r in csv.DictReader(open(sys.argv[1])))
def split(rows, gap_ns):
    out = [[]]
    for r in rows:
        if out[-1] and r[0] - out[-1][-1][1] > gap_ns:
            out.append([])
        out[-1].append(r)
    return out
seg = split(rows, 400_000_000)[-1]
if "--tail" in sys.argv:
    seg = split(seg, 20_000_000)[-1]
span = (seg[-1][1] - seg[0][0]) / 1e9
busy = sum(e - s for s, e, _, _ in seg) / 1e9
print("%d launches, %.3f s span, %.3f s in kernels, %.3f s between them" % (len(seg), span, busy, span - busy))
totals = {}
for s, e, name, grid in seg:
    c, t = totals.get((name, grid), (0, 0))
    totals[(name, grid)] = (c + 1, t + e - s)
print("%-12s %6s %9s %9s %s" % ("grid", "calls", "total ms", "mean us", "kernel"))
for (name, grid), (c, t) in sorted(totals.items(), key=lambda kv: -kv[1][1])[:25]:
    print("%-12s %6d %9.1f %9.1f %s" % (grid, c, t / 1e6, t / 1e3 / c, name[:110]))
PY
our_table() {
    # the decoder server's trace is the largest whose kernels are h3d_* (the selftest's 1024² is smaller)
    trace=$(ls -S $(grep -l '"h3d_' /work/out/h3-image-bench-trace/cupti-trace.csv.*) | head -1)
    echo "=== $1: warm 2048² decode kernels ==="
    python3 /work/decoder_kernels.py "$trace"
}
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_SKIP_COMPARE=1 sh docs/recipes/minimax-h3-cuda/bench-trace-job.sh
git -c safe.directory='*' apply "$patch_file"
our_table "main's decoder"
H3_BENCH_SKIP_COMPARE=1 sh docs/recipes/minimax-h3-cuda/bench-trace-job.sh
our_table "this commit's decoder"
sh docs/recipes/minimax-h3-cuda/mmh3-server-job.sh
echo "=== mmh3: warm picture's VAE (after the last >20 ms gap) ==="
python3 /work/decoder_kernels.py "$(ls -S /work/out/mmh3-server/cupti-trace.csv.* | head -1)" --tail
echo "decoder-trace-job done"
