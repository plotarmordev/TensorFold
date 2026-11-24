#!/bin/bash
# bench.sh LABEL BINARY [serve flags]: one server session, bench_concurrent at 1 and 4 streams, greedy.
# Inside the Desk's Mac GPU window only, one run at a time (-t 0: a held lock means wait and retry, not queue):
#   timeout 600 lockf -t 0 -k /tmp/tensorfold-model.lock docs/recipes/inkling-2-mini-metal/bench.sh c1plain BIN --parallel 4
# `--drafter "$HOME/models/Inkling-2-Mini"` (the checkpoint's own mtp.safetensors). Main decodes about 1.16 s a
# round, so control runs take TOKENS=32 to fit the 600 s. A speed run starts only when the 1-minute load is under
# LOAD_MAX (default 10; the Desk's noisy-median rule), checked each minute, and guard.py stops a void run: MIN_TPS
# is a floor on every cell's aggregate tok/s (default 0.3, below main's 0.85-0.9 control baseline; the candidate
# takes ~20), any cell with failed or unmeasured requests fails, and PARTNER_JSON=<first control's .json> on the
# second control run aborts a control more than 2x under its partner. Controls pair: a control legitimately sits
# tens of times under the candidate, so the candidate takes only the floor.
set -euo pipefail
label=$1; bin=$2; shift 2
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=${OUT:-$wt/.scratch/ink2-bench}
mkdir -p "$out"
window=/Users/Shared/agent-work/mac-gpu-window
[ -e "$window" ] || { echo "no Mac GPU window" >&2; exit 3; }
load_max=${LOAD_MAX:-10}
while :; do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk -v l="$load" -v m="$load_max" 'BEGIN {exit !(l < m)}' && break
    echo "load $load >= $load_max; waiting a minute" >&2
    sleep 60
done
# The wait can outlive the window: re-check it before the server starts.
[ -e "$window" ] || { echo "Mac GPU window closed during the load wait" >&2; exit 3; }
"$bin" serve "$HOME/models/Inkling-2-Mini" --port 18261 --context 65536 "$@" >"$out/$label.server.log" 2>&1 &
pid=$!
trap 'kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for _ in $(seq 240); do curl -sf http://127.0.0.1:18261/v1/models >/dev/null && break; sleep 1; done
TMPDIR="$out" /opt/homebrew/bin/python3.13 "$wt/tools/bench_concurrent.py" http://127.0.0.1:18261 Inkling-2-Mini \
    --levels 1,4 --reps "${REPS:-3}" --tokens "${TOKENS:-256}" --temperatures 0 --label "$label" --alone --serial \
    --output "$out/$label.json" 2>&1 | tail -n 6
MIN_TPS=${MIN_TPS:-0.3} /opt/homebrew/bin/python3.13 \
    "$wt/docs/recipes/inkling-2-mini-metal/guard.py" "$out/$label.json" ${PARTNER_JSON:+"$PARTNER_JSON"}
