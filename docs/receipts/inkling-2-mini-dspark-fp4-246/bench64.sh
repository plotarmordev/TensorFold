#!/bin/bash
# bench64.sh LABEL [--drafter DIR]: one server session at --parallel 64, bench_concurrent at 64 streams, greedy, 1 rep.
# Run it under `lockf -k /tmp/tensorfold-model.lock`, one at a time.
set -euo pipefail
label=$1; shift
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
out=${OUT:-/tmp/ink2-bench}
mkdir -p "$out"
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
"$wt/zig-out/native/bin/tensorfold-native" serve "$model" --port 18210 --context 8192 --parallel 64 "$@" >"$out/$label.server.log" 2>&1 &
pid=$!
trap 'kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 240); do curl -sf http://127.0.0.1:18210/v1/models >/dev/null && break; sleep 1; done
/opt/homebrew/bin/python3.13 "$wt/tools/bench_concurrent.py" http://127.0.0.1:18210 Inkling-2-Mini \
    --levels 64 --reps 1 --tokens 256 --temperatures 0 --label "$label" --output "$out/$label.json" 2>&1 | tail -n 4
