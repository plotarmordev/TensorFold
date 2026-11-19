#!/bin/sh
# The decoder's selftest on its tensor-core path alone, in a short TOP window: the sm_121 build, then
# `tf-h3-image-decoder-cuda selftest --tc` against the float64 host reference.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/selftest-tc.sh"
set -eu
commit=$(git -c safe.directory='*' rev-parse HEAD)
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/work/out/tensorfold-cuda-$commit/tf-h3-image-decoder-cuda selftest --tc
