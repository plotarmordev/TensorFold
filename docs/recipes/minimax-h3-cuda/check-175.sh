#!/bin/sh
# Card 175's kernel checks on TOP's Spark: h3d_flash's contract gates 1-4 (cuda-build.sh's H3_BUILD=contract
# path), then the five SMs' limits audit, whose RECORD line carries h3d_flash's registers and shared bytes:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-175.sh
set -eu
here=docs/recipes/minimax-h3-cuda
status=0
H3_BUILD=contract H3_CONTRACT=h3d_flash sh $here/cuda-build.sh || status=$?
H3_BUILD=limits sh $here/cuda-build.sh || status=$?
grep -E "h3d_flash" /work/out/limits.log || status=1
exit "$status"
