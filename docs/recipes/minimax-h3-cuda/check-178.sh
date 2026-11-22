#!/bin/sh
# Card 178's fused qkv byte-replacement gates and launch-limit receipt.
set -eu
H3_BUILD=contract H3_CONTRACT='tma heads' sh docs/recipes/minimax-h3-cuda/cuda-build.sh
H3_BUILD=limits sh docs/recipes/minimax-h3-cuda/cuda-build.sh
