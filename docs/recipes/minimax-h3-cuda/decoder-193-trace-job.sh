#!/bin/sh
# Card 193: where the default decoder stage's warm 2048² second goes, kernel by kernel. bench-trace-job.sh's run
# under the CUPTI preload, then the decoder server's last warm decode (its trace's last segment past a >400 ms idle
# gap) by stage — the VAE decoder is a ViT, so products, attention, norms and elementwise, no convolutions — and the
# idle time between kernels, which is the host's share of the banded passes.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/decoder-193-trace-job.sh
set -eu
H3_BENCH_SKIP_COMPARE=1 sh docs/recipes/minimax-h3-cuda/bench-trace-job.sh
trace=$(ls -S $(grep -l '"h3d_' /work/out/h3-image-bench-trace/cupti-trace.csv.*) | head -1)
python3 - "$trace" <<'PY'
import csv, sys
rows = sorted((int(r["start_ns"]), int(r["end_ns"]), r["name"]) for r in csv.DictReader(open(sys.argv[1])))
segs = [[]]
for r in rows:
    if segs[-1] and r[0] - segs[-1][-1][1] > 400_000_000:
        segs.append([])
    segs[-1].append(r)
seg = segs[-1]
span = (seg[-1][1] - seg[0][0]) / 1e9
busy = sum(e - s for s, e, _ in seg) / 1e9
gaps = sorted(((b[0] - a[1]) / 1e6 for a, b in zip(seg, seg[1:])), reverse=True)
print("warm decode: %d launches, %.3f s span, %.3f s in kernels, %.3f s idle; largest gaps ms %s"
      % (len(seg), span, busy, span - busy, " ".join("%.1f" % g for g in gaps[:12])))
def stage(name):
    for key, label in (("h3d_flash", "attention"), ("h3d_heads", "heads (q k norm, rope)"),
                       ("h3d_res_norm", "norms"), ("h3d_rms_norm", "norms"), ("h3d_layer_norm", "norms"),
                       ("h3d_finish", "finish"), ("h3d_swiglu", "swiglu"), ("h3d_post_quant", "embedding"),
                       ("h3d_embed", "embedding")):
        if key in name:
            return label
    return "products (cuBLASLt)"
totals = {}
for s, e, name in seg:
    for key in (stage(name), "  " + name[:100]):
        c, t = totals.get(key, (0, 0))
        totals[key] = (c + 1, t + e - s)
print("%6s %9s %s" % ("calls", "total ms", "stage / kernel"))
for key, (c, t) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
    print("%6d %9.1f %s" % (c, t / 1e6, key))
PY
echo "decoder-193-trace-job done"
