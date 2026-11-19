#!/bin/sh
# Card 130's same-job A/B (job clones are too shallow for an in-checkout git revert): the bench runs on
# this commit — the fused --tc decoder — then bench-130-base.patch is reverse-applied to put the card's
# files back to main's and the bench runs again on the default --tc path. Results: /work/out/h3-image-bench
# (this commit) and /work/out/h3-image-bench-base (main's decoder).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-130-ab.sh"
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-130-base.patch
status=0
echo "=== after: this commit ==="
H3_BENCH_OUT=/work/out/h3-image-bench sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
echo "=== before: main's --tc decoder (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_BENCH_SKIP_COMPARE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
exit "$status"
