#!/bin/sh
# Card 120's A/B pair in one job: main's bench-script.sh first, then this branch's, each suite's
# /work/out/h3-image-bench kept under card120/{main,heads-fused}. The door's checkout is a detached sha with
# no remote refs, so main is the branch's base commit: CARD120_MAIN overrides it when the branch moves.
set -eu
base=${CARD120_MAIN:-a55829e5a}
out=/work/out/h3-image-bench
keep=/work/out/card120
rm -rf "$keep"
mkdir -p "$keep"

# main's own bench script, so the baseline is exactly what upstream runs
git -c safe.directory='*' worktree add --detach /work/main-120 "$base"
(cd /work/main-120 && sh docs/recipes/minimax-h3-cuda/bench-script.sh)
mv "$out" "$keep/main"

sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/heads-fused"
