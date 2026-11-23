#!/bin/sh
# The h3 image decoder and prompt kernels' contract gates 1-4 on TOP (the suites check-job.sh's "h3 int8" misses),
# and drama3's d3_cfg_mix, which refused gate 1's rows on the N1X laptop:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/contract-job.sh"
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
H3_BUILD=contract H3_CONTRACT="h3d h3p d3_cfg_mix" exec sh "$here/cuda-build.sh"
