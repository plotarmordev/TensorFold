#!/bin/sh
# The int8 kernels' contract gates, then serve-check.sh's served-against-fresh latents, as one TOP job (~15 min):
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-job.sh"
set -eu
H3_BUILD=contract sh docs/recipes/minimax-h3-cuda/cuda-build.sh
sh docs/recipes/minimax-h3-cuda/serve-check.sh
