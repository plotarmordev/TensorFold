#!/bin/sh
# bench-script.sh with the transformer's first three steps on the half-size canvas
# (`tf-h3-image-cuda --draft-steps 3`): mmh3's fewer-steps path, its own TOP job beside the
# default run's. The Desk's door passes no environment, so this names the setting.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-drafts.sh"
H3_DECODER=tc H3_DRAFT_STEPS=3 exec sh "$(dirname -- "$0")/bench-script.sh"
