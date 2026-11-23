#!/bin/bash
# engines.sh TENSORFOLD_BINARY: the 1, 4 and 64-stream greedy receipt against oMLX on this Mac, 3 reps each, both
# servers admitting 64 requests. oMLX serves its own 4-bit MTP build (Inkling-2-Mini-oQ4e-mtp) from
# ~/.omlx/settings.json, MTP drafting on. Run it under `lockf -k /tmp/tensorfold-model.lock`; the two servers run one
# after the other. OMLX names the oMLX binary. 64 lanes take a 16384-token window to fit in memory.
set -euo pipefail
bin=$1
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
omlx=${OMLX:-$HOME/work/inkling/.omlx/bin/omlx}
out=${OUT:-/tmp/ink2-bench}
mkdir -p "$out"
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }

measure() { # LABEL PORT MODEL COMMAND...: start the server, bench it, stop it
    local label=$1 port=$2 name=$3
    shift 3
    "$@" >"$out/$label.server.log" 2>&1 &
    local pid=$!
    trap "kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null" EXIT
    for _ in $(seq 300); do curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 1; done
    /opt/homebrew/bin/python3.13 "$wt/tools/bench_concurrent.py" "http://127.0.0.1:$port" "$name" \
        --levels 1,4,64 --reps 3 --tokens 256 --temperatures 0 --label "$label" --output "$out/$label.json" 2>&1 | cut -c1-200
    kill -INT $pid
    wait $pid 2>/dev/null || true
    trap - EXIT
}

measure tensorfold 18235 Inkling-2-Mini "$bin" serve "$model" --port 18235 --context 16384 --parallel 64 --drafter "$model"
measure omlx 8765 Inkling-2-Mini-oQ4e-mtp "$omlx" serve --host 127.0.0.1 --port 8765 --max-concurrent-requests 64
