#!/bin/sh
# mmh3-server-job.sh with draft-steps 3 on every 2048² request (mmh3's `POST /v1/images` takes the
# field of the same name): the warm-draft seconds and PNGs beside bench-drafts.sh's own job. The
# Desk's door passes no environment, so this names the setting like bench-drafts.sh does.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/mmh3-drafts-job.sh"
set -eu
MMH3_DRAFT_STEPS=3 exec sh "$(dirname -- "$0")/mmh3-server-job.sh"
