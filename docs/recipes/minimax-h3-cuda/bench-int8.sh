#!/bin/sh
# bench-script.sh with the transformer's projections as block-scaled int8 (`tf-h3-image-cuda --int8`): the Desk's
# door passes no environment, so this names the setting.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-int8.sh"
H3_LINEAR=int8 exec sh "$(dirname -- "$0")/bench-script.sh"
