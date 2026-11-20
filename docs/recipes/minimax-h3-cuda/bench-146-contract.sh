#!/bin/sh
# Card 146's contract + bench on TOP's Spark: builds this commit's tf-cuda-test for sm_121 through
# cuda-build.sh's H3_BUILD=contract path, runs gates 1-4 on the int8 kernels (the TMA product and
# h3_quantize), then benches the TMA int8 product against cuBLASLt int8 on the picture's four projections:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-146-contract.sh"
set -eu
status=0
H3_BUILD=contract H3_CONTRACT=int8 sh docs/recipes/minimax-h3-cuda/cuda-build.sh || status=$?
/work/contract/bin/tf-cuda-test h3-int8-bench
exit "$status"
