#!/bin/bash
# run.sh CONTROL_TREE: card 234's window sequence under one lock. CONTROL_TREE is a checkout of the control commit with
# its own `zig build native -Doptimize=ReleaseFast`; this tree is the candidate. oMLX's cells come from card 235's
# engines.sh; MLX_PYTHON is a Python with MLX for the bandwidth ceiling. Run it as
# `lockf -k /tmp/tensorfold-model.lock run.sh CONTROL_TREE`.
set -euo pipefail
ctrl=$1
wt=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
mlx=${MLX_PYTHON:-$HOME/models/mlxref/bin/python3}
r=docs/receipts/inkling-2-mini-metal-210
step() { echo "== $1 $(date +%T) load: $(sysctl -n vm.loadavg)"; }
step bandwidth; "$mlx" "$wt/docs/receipts/inkling-2-mini-metal-234/bandwidth.py"
step ctrl1ds; "$ctrl/$r/bench.sh" ctrl1ds --drafter "$model"
step cand1ds; "$wt/$r/bench.sh" cand1ds --drafter "$model"
step ctrl2ds; "$ctrl/$r/bench.sh" ctrl2ds --drafter "$model"
step cand2ds; "$wt/$r/bench.sh" cand2ds --drafter "$model"
step ctrl1plain; "$ctrl/$r/bench.sh" ctrl1plain
step candplain; "$wt/$r/bench.sh" candplain
step ctrl2plain; "$ctrl/$r/bench.sh" ctrl2plain
step prefill; "$wt/$r/prefill.sh" candprefill --drafter "$model"
step engines; "$wt/docs/receipts/inkling-2-mini-metal-234/engines.sh"
step done
