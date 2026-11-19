#!/bin/sh
# bench-script.sh with the transformer's projections as block-scaled FP8 (`tf-h3-image-cuda --fp8`): the Desk's door
# passes no environment, so this names the setting.
H3_LINEAR=fp8 exec sh docs/recipes/minimax-h3-cuda/bench-script.sh
