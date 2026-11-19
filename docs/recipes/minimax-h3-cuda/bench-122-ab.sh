#!/bin/sh
# Card 122's same-job A/B (job clones are too shallow for an in-checkout git revert): the bench runs on
# this commit, then bench-122-base.patch is reverse-applied to put the card's h3 files back to main's
# and the bench runs again with H3_DECODER=tc — the baseline is main's decoder with --tc passed.
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main + --tc).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-122-ab.sh"
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-122-base.patch
status=0
echo "=== after: this commit ==="
H3_BENCH_OUT=/work/out/h3-image-bench sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
echo "=== before: main + --tc (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_DECODER=tc H3_BENCH_SKIP_COMPARE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
exit "$status"
