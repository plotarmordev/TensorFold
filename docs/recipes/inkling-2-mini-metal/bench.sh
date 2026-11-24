#!/bin/bash
# bench.sh LABEL BINARY [serve flags]: one server session, bench_concurrent at 1 and 4 streams, greedy.
# Inside the Desk's Mac GPU window only, one run at a time (-t 0: a held lock means wait and retry, not queue):
#   timeout 600 lockf -t 0 -k /tmp/tensorfold-model.lock docs/recipes/inkling-2-mini-metal/bench.sh c1plain BIN --parallel 4
# Order: blit.swift (ceiling), then control (main) / candidate / control, each plain and with
# `--drafter "$HOME/models/Inkling-2-Mini"` (the checkpoint's own mtp.safetensors).
set -euo pipefail
label=$1; bin=$2; shift 2
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=${OUT:-$wt/.scratch/ink2-bench}
mkdir -p "$out"
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
"$bin" serve "$HOME/models/Inkling-2-Mini" --port 18261 --context 65536 "$@" >"$out/$label.server.log" 2>&1 &
pid=$!
trap 'kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 240); do curl -sf http://127.0.0.1:18261/v1/models >/dev/null && break; sleep 1; done
TMPDIR="$out" /opt/homebrew/bin/python3.13 "$wt/tools/bench_concurrent.py" http://127.0.0.1:18261 Inkling-2-Mini \
    --levels 1,4 --reps "${REPS:-3}" --tokens "${TOKENS:-256}" --temperatures 0 --label "$label" --alone --serial \
    --output "$out/$label.json" 2>&1 | tail -n 6
