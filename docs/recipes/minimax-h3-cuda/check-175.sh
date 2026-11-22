#!/bin/sh
# Card 175's kernel check on TOP's Spark: h3d_flash's contract gates 1-4 (cuda-build.sh's H3_BUILD=contract
# path) into /work/out/contract.log, its gate lines past the Desk's 200 log lines. Its registers and shared
# bytes come from cuda-build.sh's own H3_BUILD=limits job (the RECORD line).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-175.sh
set -eu
status=0
H3_BUILD=contract H3_CONTRACT=h3d_flash sh docs/recipes/minimax-h3-cuda/cuda-build.sh >/work/out/contract.log 2>&1 || status=$?
grep -vE "warning|Remark|^ *\^|detected during|^ *$|constexpr|^ *(\+-|kernel-contract-fatbins)" /work/out/contract.log | tail -n 60
echo "contract exit $status"
exit "$status"
