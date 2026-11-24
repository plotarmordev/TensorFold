#!/bin/bash
# run.sh CONTROL_TREE: card 262's Mac GPU window. CONTROL_TREE is a checkout of the control (card 248's head,
# 1cb6ac2a9); this tree is the candidate. The DSpark device tests and the float64 gate run first and stop the window
# on a failure, then both trees build and control/candidate/control measure at 1 and 4 streams with 248's bench.sh.
# The device tests take /tmp/tensorfold-model.lock themselves, so the gates run outside the lock and the benches
# inside it; run from the candidate tree in the foreground while /Users/Shared/agent-work/mac-gpu-window exists.
set -euo pipefail
ctrl=$1
here=$(cd "$(dirname "$0")" && pwd)
wt=$(git -C "$here" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
bench=$wt/docs/receipts/inkling-2-mini-metal-248/bench.sh
export OUT=${OUT:-$wt/.scratch/ink2-bench-262}
mkdir -p "$OUT"
step() { echo "== $1 $(date +%T) load: $(sysctl -n vm.loadavg)"; }
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
cd "$wt"
step device-tests; TF_INKLING2_METAL_DIR=$model TF_INKLING2_DSPARK_DIR=$model zig build test-inkling2-metal-host
step float64; TF_INKLING2_METAL_DIR=$model zig build test-inkling2-metal-float64
step build-control; (cd "$ctrl" && zig build native -Doptimize=ReleaseFast)
step build-candidate; zig build native -Doptimize=ReleaseFast
step measure-ctrl1; lockf -k /tmp/tensorfold-model.lock "$bench" "$ctrl" ctrl1 --drafter "$model"
step measure-cand; lockf -k /tmp/tensorfold-model.lock "$bench" "$wt" cand --drafter "$model"
step measure-ctrl2; lockf -k /tmp/tensorfold-model.lock "$bench" "$ctrl" ctrl2 --drafter "$model"
step done
