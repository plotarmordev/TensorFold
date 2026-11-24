#!/bin/bash
# bench.sh LABEL BINARY: one DSpark server session from BINARY admitting 16 lanes, bench_concurrent at 1, 2, 4 and 16
# streams, greedy, 3 reps. Run it under `lockf -k /tmp/tensorfold-model.lock`, one at a time; LEVELS overrides the
# stream counts.
set -euo pipefail
label=$1 bin=$2
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
out=${OUT:-/tmp/ink2-bench}
mkdir -p "$out"
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
"$bin" serve "$model" --port 18235 --context 65536 --parallel 16 --drafter "$model" >"$out/$label.server.log" 2>&1 &
pid=$!
trap 'kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 240); do curl -sf http://127.0.0.1:18235/v1/models >/dev/null && break; sleep 1; done
/opt/homebrew/bin/python3.13 "$wt/tools/bench_concurrent.py" http://127.0.0.1:18235 Inkling-2-Mini \
    --levels "${LEVELS:-1,2,4,16}" --reps 3 --tokens 256 --temperatures 0 --label "$label" --alone \
    --output "$out/$label.json" 2>&1 | tail -n 6
