#!/bin/sh
# Card 144 in one TOP window: the decoder kernel tables (main, this commit, mmh3's VAE), then the A/B.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/decoder-144-job.sh
set -eu
status=0
sh docs/recipes/minimax-h3-cuda/decoder-trace-job.sh || status=$?
sh docs/recipes/minimax-h3-cuda/bench-144-ab.sh || status=$?
exit "$status"
