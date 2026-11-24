#!/bin/bash
# engines.sh: TensorFold with DSpark at 1, 4 and 64 streams, greedy, 3 reps, on 64 lanes (the default 8 queues the rest).
# Card 235's engines.sh runs oMLX the same way. Run it under `lockf -k /tmp/tensorfold-model.lock`.
set -euo pipefail
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
out=${OUT:-/tmp/ink2-bench}
mkdir -p "$out"
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
"$wt/zig-out/native/bin/tensorfold-native" serve "$model" --port 18234 --context 65536 --parallel 64 --drafter "$model" \
    >"$out/tensorfold64.server.log" 2>&1 &
pid=$!
trap 'kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 300); do curl -sf http://127.0.0.1:18234/v1/models >/dev/null && break; sleep 1; done
/opt/homebrew/bin/python3.13 "$wt/tools/bench_concurrent.py" http://127.0.0.1:18234 Inkling-2-Mini \
    --levels 1,4,64 --reps 3 --tokens 256 --temperatures 0 --label tensorfold64 --output "$out/tensorfold64.json" 2>&1 | cut -c1-120
