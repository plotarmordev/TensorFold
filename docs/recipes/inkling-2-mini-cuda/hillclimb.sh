#!/bin/sh
# Inkling 2 Mini decode levers on TOP's GB10, the inference-optimisation skill's way:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/hillclimb.sh
# After setup.sh's build and exactness gates: the measured copy bandwidth and the decode ceilings at 1 and 64
# streams, then fresh servers in the order control, lever A, control, lever B, control, each benched at 1 stream
# (6 reps) and 64 streams (3 reps), greedy, 128 tokens. A lever is kept only when its median total tok/s beats both
# neighbouring controls at its level and loses no more than 3% at the other; any failed request stops the job.
# Every arm's numbers land in /work/out/inkling2/ledger.jsonl.
set -eu
. docs/recipes/inkling-2-mini-cuda/setup.sh
"$bin/tf-cuda-test" carveout >"$out/carveout.log" 2>&1 || :
cat "$out/carveout.log"
gbps=$(sed -n 's/.*device->device \([0-9]*\);.*/\1/p' "$out/carveout.log")
echo "measured device->device copy: ${gbps:-unmeasured} GB/s"
python3 docs/recipes/inkling-2-mini-cuda/ceiling.py "$model" "${gbps:-273}" 1 64
arm() {
    name=$1
    shift
    env "$@" /work/sm121/native/bin/tensorfold-native serve "$model" --backend cuda --host 127.0.0.1 --port 8188 \
        --name inkling-2-mini --parallel 64 --context 4096 --no-update-check >"$out/server-$name.log" 2>&1 &
    server=$!
    python3 - "$server" <<'EOF'
import os, sys, time, urllib.request
for _ in range(900):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)
        raise SystemExit(0)
    except OSError:
        # the server is the shell's child, not ours: a zombie or a gone pid has exited
        try:
            state = open(f"/proc/{sys.argv[1]}/stat").read().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            state = "Z"
        if state == "Z":
            raise SystemExit("the server exited before answering")
        time.sleep(1)
raise SystemExit("server did not answer /v1/models in 15 min")
EOF
    for level in 1 64; do
        reps=$([ "$level" = 1 ] && echo 6 || echo 3)
        python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels "$level" --tokens 128 \
            --reps "$reps" --temperatures 0 --label "$name-$level" --output "$out/$name-$level.json" >/dev/null
    done
    kill -INT "$server"
    wait "$server" || :
    python3 - "$out" "$name" "$*" <<'EOF' | tee -a "$out/ledger.jsonl"
import json, statistics, sys
out, name, env = sys.argv[1:]
row = {"arm": name, "env": env}
for level in (1, 64):
    cells = json.load(open(f"{out}/{name}-{level}.json"))["cells"]
    if any(c["failed"] for c in cells):
        raise SystemExit(f"{name} at {level} streams: a request failed")
    row[str(level)] = round(statistics.median(r["aggregate_tps"] for c in cells for r in c["reps"]), 1)
print(json.dumps(row))
EOF
}
arm control-1
arm warps-1 TF_NVFP4_DECODE_WARPS=1
arm control-2
arm half-0 TF_QMMF_HALF_BELOW=0
arm control-3
python3 - "$out/ledger.jsonl" <<'EOF'
import json, sys
rows = {r["arm"]: r for r in map(json.loads, open(sys.argv[1]))}
for lever, (a, b) in {"warps-1": ("control-1", "control-2"), "half-0": ("control-2", "control-3")}.items():
    c, before, after = rows[lever], rows[a], rows[b]
    gains = {k: c[k] / max(before[k], after[k]) - 1 for k in ("1", "64")}
    worst = {k: c[k] / min(before[k], after[k]) - 1 for k in ("1", "64")}
    kept = any(g > 0 for g in gains.values()) and all(w > -0.03 for w in worst.values())
    print(f"RESULT inkling2 mini lever {lever} ({c['env']}): 1 stream {c['1']} vs {before['1']}/{after['1']}, "
          f"64 streams {c['64']} vs {before['64']}/{after['64']} ({gains['64']:+.1%} over the better control): "
          f"{'kept' if kept else 'rejected'}")
EOF
