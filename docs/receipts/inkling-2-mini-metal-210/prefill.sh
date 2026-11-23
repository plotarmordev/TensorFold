#!/bin/bash
# prefill.sh LABEL [server args]: one locked session, three cold 45K prompts and their cached repeats.
set -euo pipefail
label=$1; shift
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=${OUT:-/tmp/ink2-bench}
mkdir -p "$out"
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
"$wt/zig-out/native/bin/tensorfold-native" serve "${MODEL:-$HOME/models/Inkling-2-Mini}" --port 18210 --context 65536 "$@" \
    >"$out/$label.server.log" 2>&1 &
pid=$!
trap 'kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 240); do curl -sf http://127.0.0.1:18210/v1/models >/dev/null && break; sleep 1; done
/opt/homebrew/bin/python3.13 "$(dirname "$0")/prefill.py" http://127.0.0.1:18210 "$wt/LICENSE" | tee "$out/$label.txt"
