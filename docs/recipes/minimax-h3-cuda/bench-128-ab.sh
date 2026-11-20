#!/bin/sh
# Card 128's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-128-base.patch reverse-applied — the card's six files back to main's — and
# bench-script.sh again: the baseline is the same job's main code with this commit's bench script,
# model paths and --tc decoder default.
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (the card's files at main).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-128-ab.sh"
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-128-base.patch
status=0
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
H3_PROFILE=1 sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
echo "=== before: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_BENCH_SKIP_COMPARE=1 H3_PROFILE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
exit "$status"
