#!/bin/bash
# run.sh CONTROL_TREE: card 248's Mac GPU window in two phases under one lock. CONTROL_TREE is a checkout of the
# control commit (the base branch's tip) with its own `zig build native -Doptimize=ReleaseFast`; this tree is the
# candidate. Build first, then measure control/candidate/control at 1 and 4 streams. Run it as
# `lockf -k /tmp/tensorfold-model.lock docs/receipts/inkling-2-mini-metal-248/run.sh CONTROL_TREE`.
set -euo pipefail
ctrl=$1
here=$(dirname "$0")
wt=$(git -C "$here" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
step() { echo "== $1 $(date +%T) load: $(sysctl -n vm.loadavg)"; }
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
step build-control; (cd "$ctrl" && zig build native -Doptimize=ReleaseFast)
step build-candidate; (cd "$wt" && zig build native -Doptimize=ReleaseFast)
step submission-cost; swift "$here/submission.swift"
step measure-ctrl1; "$here/bench.sh" "$ctrl" ctrl1 --drafter "$model"
step measure-cand; "$here/bench.sh" "$wt" cand --drafter "$model"
step measure-ctrl2; "$here/bench.sh" "$ctrl" ctrl2 --drafter "$model"
step done
