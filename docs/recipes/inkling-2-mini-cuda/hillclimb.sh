#!/bin/sh
# Inkling 2 Mini decode levers on TOP's GB10, the inference-optimisation skill's way:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/hillclimb.sh
# After setup.sh's build and exactness gates, estimate the no-draft ceilings at 1 and 64 streams, then bench fresh
# servers control, lever A, control, lever B, control. Each arm runs same-prompt c1 (6 reps), same-prompt c64 and
# mixed-prompt c64 (3 reps each), greedy, 128 tokens. Every stream must produce a valid timing. Keep a lever only if
# its median beats both neighbouring controls by over 3% at one level/case without losing over 3% elsewhere.
# Every successful or failed arm lands in /work/out/inkling2/ledger.jsonl.
set -eu
if [ "${INKLING_HILLCLIMB_DEADLINE:-}" != 1 ]; then
    INKLING_HILLCLIMB_DEADLINE=1 exec timeout --signal=INT --kill-after=20s 580 sh "$0"
fi
. docs/recipes/inkling-2-mini-cuda/setup.sh
python3 docs/recipes/inkling-2-mini-cuda/ceiling.py "$model" 273 1 64
prompts=docs/recipes/inkling-2-mini-cuda/prompts-64.json
prompt_sha=$(sha256sum "$prompts" | cut -d' ' -f1)
server=
stop_server() {
    if [ -n "$server" ]; then
        kill -INT "$server" 2>/dev/null || :
        wait "$server" 2>/dev/null || :
        server=
    fi
}
trap 'stop_server' EXIT
trap 'stop_server; exit 130' INT
trap 'stop_server; exit 143' TERM
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
    failed_case=
    for case in same-1 same-64 mixed-64; do
        level=${case##*-}
        reps=$([ "$level" = 1 ] && echo 6 || echo 3)
        if [ "$case" = mixed-64 ]; then
            python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels 64 --tokens 128 \
                --reps "$reps" --temperatures 0 --prompts "$prompts" --label "$name-$case" \
                --output "$out/$name-$case.json" >/dev/null || failed_case=$case
        else
            python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels "$level" --tokens 128 \
                --reps "$reps" --temperatures 0 --label "$name-$case" --output "$out/$name-$case.json" >/dev/null || failed_case=$case
        fi
        [ -z "$failed_case" ] || break
    done
    stop_server
    if [ -n "$failed_case" ]; then
        printf '{"arm":"%s","case":"%s","failed":"benchmark exited nonzero","prompts_file":"%s","prompts_sha256":"%s"}\n' \
            "$name" "$failed_case" "$prompts" "$prompt_sha" | tee -a "$out/ledger.jsonl"
        return 1
    fi
    python3 - "$out" "$name" "$*" "$prompts" "$prompt_sha" <<'EOF'
import json, math, statistics, sys
out, name, env, prompts, prompt_sha = sys.argv[1:]
row = {"arm": name, "env": env, "pack": "thinkingmachines/Inkling-2-Mini ed40e7da62",
       "prompts_file": prompts, "prompts_sha256": prompt_sha}
bad = []
for case in ("same-1", "same-64", "mixed-64"):
    report = json.load(open(f"{out}/{name}-{case}.json"))
    cells = report["cells"]
    level = int(case.split("-")[1])
    reps = 6 if level == 1 else 3
    if len(cells) != (1 if case == "mixed-64" else 2) or any(
           c["streams"] != level or c["failed"] or c["unmeasured"] or len(c["reps"]) != reps or
           any(r["failed"] or r["unmeasured"] or len(r["per_stream_tps"]) != level or
               len(r["requests"]) != level or not math.isfinite(r["aggregate_tps"]) or
               r["aggregate_tps"] <= 0 for r in c["reps"]) for c in cells):
        bad.append(case)
        row[case] = {"failed": sum(c["failed"] for c in cells),
                     "unmeasured": sum(c["unmeasured"] for c in cells)}
    else:
        row[case] = round(statistics.median(r["aggregate_tps"] for c in cells for r in c["reps"]), 1)
with open(f"{out}/ledger.jsonl", "a") as ledger:
    ledger.write(json.dumps(row) + "\n")
print(json.dumps(row))
if bad:
    raise SystemExit(f"{name}: invalid measurements in {', '.join(bad)}")
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
    gains = {k: c[k] / max(before[k], after[k]) - 1 for k in ("same-1", "same-64", "mixed-64")}
    worst = {k: c[k] / min(before[k], after[k]) - 1 for k in gains}
    kept = any(g > 0.03 for g in gains.values()) and all(w > -0.03 for w in worst.values())
    print(f"RESULT inkling2 mini lever {lever} ({c['env']}): " + ", ".join(
          f"{k} {c[k]} vs {before[k]}/{after[k]} ({gains[k]:+.1%} over better control)" for k in gains) +
          f": {'kept' if kept else 'rejected'}")
EOF
