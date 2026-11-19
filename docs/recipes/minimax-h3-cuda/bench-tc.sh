#!/bin/sh
# bench-script.sh with the decoder's products and attention on tensor cores (--tc): its own TOP job,
# beside the default run's, for the same-latent PSNR and warm-wall comparison.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-tc.sh"
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
H3_DECODER=tc exec sh "$here/bench-script.sh"
