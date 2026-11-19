#!/bin/sh
# Card 129's contract + microbench job: the rebuilt h3 int8 tile's gates 1-4, then the tile-vs-cuBLASLt
# A/B on the four picture shapes (`h3-int8-bench`). The card's stop condition: if the tile does not beat
# cuBLASLt's int8 rate, stop before any picture run.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-129-contract.sh"
set -eu
export H3_BUILD=contract
sh docs/recipes/minimax-h3-cuda/cuda-build.sh
BIN=/work/contract/bin/tf-cuda-test
"$BIN" h3-int8-bench | tee /work/out/h3-int8-bench.txt
