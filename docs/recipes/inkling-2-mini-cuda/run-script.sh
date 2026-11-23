#!/bin/sh
# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, revision ed40e7da62) whole on TOP's GB10:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/run-script.sh
# The job has no network: the Desk put the checkpoint in the pilot folder as Inkling-2-Mini, and the toolchain
# archives of ../minimax-h3-cuda/toolchain-stage.sh beside it. Logs and the 1/4-stream receipt land in /work/out/inkling2.
set -eu
. docs/recipes/inkling-2-mini-cuda/setup.sh
mem() { echo "$1: host $(awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2} END {printf "used %d MiB, available %d MiB", (t - a) / 1024, a / 1024}' /proc/meminfo); GPU $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; }
mem "before the server"
/work/sm121/native/bin/tensorfold-native serve "$model" --backend cuda --host 127.0.0.1 --port 8188 \
    --name inkling-2-mini --parallel 4 --context 8192 --no-update-check >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || :' EXIT
python3 - <<'EOF'
import time, urllib.request
for _ in range(900):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)
        raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("server did not answer /v1/models in 15 min")
EOF
mem "server loaded"
for level in 1 4; do
    python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels "$level" --tokens 256 --reps 3 \
        --temperatures 0 --label "gb10-$level" --output "$out/streams-$level.json"
    mem "after $level streams"
done
python3 - "$out" "$model" <<'EOF'
import json, sys
sys.path.insert(0, "docs/recipes/inkling-2-mini-cuda")
from ceiling import round_bytes
out, model = sys.argv[1:]
agg = {}
for level in (1, 4):
    cells = json.load(open(f"{out}/streams-{level}.json"))["cells"]
    if any(c.get("failed") for c in cells):
        raise SystemExit(f"{level} streams: a request failed")
    agg[level] = max(c["aggregate_tps"] for c in cells)
    for c in cells:
        r = c["request"]
        print(f"RESULT inkling2 mini gb10 {level} stream(s) {c['prompt']}: first token {r['ttft_s']} s, prompt {r['prompt_ttft_tps']} tok/s, decode {c['per_stream_tps']} tok/s a stream, {c['aggregate_tps']} total")
per_token = round_bytes(model, 1)
bandwidth = 273e9  # GB10's LPDDR5x peak, NVIDIA's DGX Spark specification
print(f"RESULT inkling2 mini gb10 aggregate decode: 1 stream {agg[1]:.1f} tok/s, 4 streams {agg[4]:.1f} tok/s ({agg[4] / agg[1]:.2f}x)")
print(f"RESULT inkling2 mini weight bytes a token {per_token / 1e9:.2f} GB: one stream's bandwidth limit {bandwidth / per_token:.1f} tok/s at 273 GB/s, measured {100 * agg[1] * per_token / bandwidth:.0f}% of it")
if agg[4] < 1.5 * agg[1]:
    raise SystemExit(f"4 streams gave {agg[4]:.1f} tok/s, not clearly more than 1 stream's {agg[1]:.1f}")
EOF
