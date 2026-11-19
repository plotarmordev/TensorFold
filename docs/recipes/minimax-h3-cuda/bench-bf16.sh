#!/bin/sh
# bench-script.sh with the transformer's projections in bf16 (`tf-h3-image-cuda --bf16`): the Desk's door passes no
# environment, so this names the setting.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-bf16.sh"
H3_LINEAR=bf16 exec sh "$(dirname -- "$0")/bench-script.sh"
