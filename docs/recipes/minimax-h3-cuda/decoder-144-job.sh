#!/bin/sh
# Card 144 in one TOP window: the decoder kernel tables (main, this commit, mmh3's VAE), the A/B, then
# the limits audit whose limits.log holds h3d_flash's RECORD line.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/decoder-144-job.sh
set -eu
status=0
sh docs/recipes/minimax-h3-cuda/decoder-trace-job.sh || status=$?
sh docs/recipes/minimax-h3-cuda/bench-144-ab.sh || status=$?
sh docs/recipes/minimax-h3-cuda/limits-job.sh || status=$?
grep "h3d_flash" /work/out/*/limits.log || true
exit "$status"
