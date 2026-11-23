#!/bin/sh
# Card 178's fused qkv byte-replacement gates and launch-limit receipt.
set -eu
H3_BUILD=contract H3_CONTRACT=h3_heads_int8_tma exec timeout 600 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
