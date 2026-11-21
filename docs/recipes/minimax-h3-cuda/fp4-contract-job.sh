#!/bin/sh
# Card 147's NVFP4 quantizers under the contract on TOP: gates 1-4 on h3_quantize_fp4, the fused self-test
# (h3_norm_quant_fp4 byte for byte h3_norm then h3_quantize_fp4), then the h3 image's limits, whose RECORD lines
# are the fp4 launches' measured usage:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/fp4-contract-job.sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
H3_BUILD=sm121 sh "$here/cuda-build.sh"
LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib /work/sm121/bin/tf-h3-image-cuda --self-test
status=0
H3_BUILD=contract H3_CONTRACT="h3 fp4" sh "$here/cuda-build.sh" || status=$?
echo "kernel-contract h3 fp4 exit $status"
H3_BUILD=limits sh "$here/cuda-build.sh" || true
grep -E "fp4" /work/out/limits.log
exit "$status"
