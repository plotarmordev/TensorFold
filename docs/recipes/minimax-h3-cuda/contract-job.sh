#!/bin/sh
# The h3 image decoder kernels' contract gates 1-4 on TOP — the suite check-job.sh's "h3 int8" misses:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/contract-job.sh"
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
H3_BUILD=contract H3_CONTRACT=h3d exec sh "$here/cuda-build.sh"
