#!/bin/sh
# Inkling 2 Mini's trunk (attention projections and unembedding) quantised at load, against the bf16 receipt:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/trunk-ab.sh [mxfp8|nvfp4 ...]
# Each trunk (default mxfp8 then nvfp4): top-1 and mean KL against the float64 reference over 32 greedy tokens
# (tf-cuda-test inkling2-mini-ref, its gate in the log), and decode at 1 and 4 streams with run-script.sh's settings.
# The reference runs on one host core beside the bench; the bf16 rows are run-script.sh's receipt (32.9 / 118.7).
set -eu
if [ -z "${INKLING2_TIMED:-}" ]; then
    INKLING2_BUILD_ONLY=1 sh "$(dirname "$0")/run-script.sh"
    INKLING2_TIMED=1 exec timeout 600 sh "$0" "$@"
fi
pilot=/assets/drama3-pilot
model=$pilot/Inkling-2-Mini
[ -f "$model/config.json" ] || model=$pilot/models/Inkling-2-Mini
bin=/work/sm121/bin
out=/work/out/inkling2-trunk
rm -rf "$out"
mkdir -p "$out"
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
text="The three laws of thermodynamics, explained for a curious student:"
[ $# -gt 0 ] || set -- mxfp8 nvfp4
for trunk in "$@"; do
    TF_INKLING2_TRUNK=$trunk "$bin/tf-cuda-test" inkling2-mini-ref "$model" 32 "$text" >"$out/ref-$trunk.log" 2>&1 &
    ref=$!
    TF_INKLING2_TRUNK=$trunk /work/sm121/native/bin/tensorfold-native serve "$model" --backend cuda --host 127.0.0.1 \
        --port 8188 --name inkling-2-mini --parallel 4 --context 8192 --no-update-check >"$out/server-$trunk.log" 2>&1 &
    server=$!
    trap 'kill $server $ref 2>/dev/null || :' EXIT
    python3 - <<'EOF'
import time, urllib.request
for _ in range(240):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)
        raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("server did not answer /v1/models in 4 min")
EOF
    grep "GiB of weights" "$out/server-$trunk.log" || :
    for level in 1 4; do
        python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels "$level" --tokens 256 --reps 2 \
            --temperatures 0 --label "$trunk-$level" --output "$out/streams-$trunk-$level.json" >/dev/null
        python3 -c 'import json, sys; c = json.load(open(sys.argv[1]))["cells"]; assert not any(x.get("failed") for x in c), "a request failed"; print(f"RESULT inkling2 mini {sys.argv[2]} trunk gb10 {sys.argv[3]} stream(s): {max(x[\"aggregate_tps\"] for x in c):.1f} tok/s total")' \
            "$out/streams-$trunk-$level.json" "$trunk" "$level"
    done
    kill $server
    wait $server 2>/dev/null || :
    status=0
    wait $ref || status=$?
    grep -E "^(RESULT|DIFF|INFO|FAIL)" "$out/ref-$trunk.log" || cat "$out/ref-$trunk.log"
    [ "$status" -eq 0 ] || echo "RESULT inkling2 mini $trunk trunk: the reference gate failed (exit $status)"
done
