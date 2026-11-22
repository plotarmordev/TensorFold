#!/bin/sh
# Card 179's int8 attention contract and five-SM resource audit on TOP.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-179.sh
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
status=0
H3_BUILD=contract H3_CONTRACT=h3_attention_i8 sh "$here/cuda-build.sh" || status=$?
H3_BUILD=limits sh "$here/cuda-build.sh" || status=$?
grep -E "limits h3_(heads_quant|attention_i8)" /work/out/limits.log || :
exit "$status"
