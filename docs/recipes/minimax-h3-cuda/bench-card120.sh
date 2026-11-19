#!/bin/sh
# Card 120's A/B pair in one job: origin/main's bench-script.sh first, then this branch's, each suite's
# /work/out/h3-image-bench kept under card120/{main,heads-fused}. The door's checkout is the workdir.
set -eu
out=/work/out/h3-image-bench
keep=/work/out/card120
rm -rf "$keep"
mkdir -p "$keep"

# main's own bench script, so the baseline is exactly what upstream runs
git worktree add --detach /work/main-120 origin/main
(cd /work/main-120 && sh docs/recipes/minimax-h3-cuda/bench-script.sh)
mv "$out" "$keep/main"

sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/heads-fused"
