#!/bin/sh
# Card 129's rescoped same-job A/B: the bench runs on this commit (all four projections on the cuBLASLt
# int8 path), then bench-129-cublaslt-base.patch is reverse-applied to put the two h3 files back to
# main's and the bench runs again — the baseline is main's default (cuBLASLt int8 on qkvc/fc1, bf16
# out/fc2, --tc decoder), re-measured in this job. H3_PROFILE is on: the per-stage lines are the card's
# required evidence (it also keeps the step's CUDA graph off, so the warm walls are no-graph times on
# both sides).
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-129-cublaslt.sh
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-129-cublaslt-base.patch
status=0
echo "=== after: this commit (all four int8 on cuBLASLt) ==="
H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-image-bench sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
echo "=== before: main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-image-bench-base \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
exit "$status"
