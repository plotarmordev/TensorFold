#!/bin/sh
# Card 120's A/B pair in one job: the baseline is the branch's own checkout reverted to main by
# card120-main.patch (the door's clone is a detached single-commit checkout — no origin/main, no
# history — so a reverse git apply on the four touched files reproduces main's tree exactly).
# Each suite's /work/out/h3-image-bench is kept under card120/{main,heads-fused}.
set -eu
out=/work/out/h3-image-bench
keep=/work/out/card120
patch=docs/recipes/minimax-h3-cuda/card120-main.patch
rm -rf "$keep"
mkdir -p "$keep"

git -c safe.directory='*' apply -R "$patch"
trap 'git -c safe.directory="*" apply "$patch" 2>/dev/null || :; rm -rf "$out" || :' EXIT
sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/main"

git -c safe.directory='*' apply "$patch"
sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/heads-fused"
