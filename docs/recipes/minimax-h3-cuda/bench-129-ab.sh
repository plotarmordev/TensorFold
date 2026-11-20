#!/bin/sh
# Card 129's same-job A/B (job clones are too shallow for an in-checkout git revert): the bench runs on
# this commit (all four projections on the own int8 tile), then bench-129-base.patch is reverse-applied to
# put the card's h3 files back to main's and the bench runs again — the baseline is main's default
# (cuBLASLt int8 on qkvc/fc1, bf16 out/fc2, --tc decoder).
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-129-ab.sh"
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-129-base.patch
status=0
echo "=== after: this commit ==="
H3_BENCH_OUT=/work/out/h3-image-bench sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
echo "=== before: main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_BENCH_SKIP_COMPARE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
exit "$status"
