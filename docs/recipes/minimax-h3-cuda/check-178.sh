#!/bin/sh
# Card 178's fused qkv byte-replacement gates and launch-limit receipt.
set -eu
H3_BUILD=contract H3_CONTRACT=h3_heads_int8_tma sh docs/recipes/minimax-h3-cuda/cuda-build.sh
H3_BUILD=limits H3_LIMITS_FILTER=h3 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
