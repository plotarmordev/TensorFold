#!/bin/sh
# Inkling 2 Mini's DSpark draft head (mtp.safetensors) on TOP's GB10 against its float64 host reference:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/dspark.sh
# Seven device-fed greedy drafts after 1,100 seeded context rows, every step's logits within 2% RMS of the
# reference teacher-forced on the device's picks, then the planted context-blind reference, which must go red.
set -eu
. docs/recipes/inkling-2-mini-cuda/toolchain.sh
t0=$(date +%s)
zig build install -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/sm121
t1=$(date +%s)
echo "build: $((t1 - t0)) s"
timeout 600 sh -c '"$1/tf-cuda-test" inkling2-dspark "$2" && INKLING2_REF_PLANT=1 "$1/tf-cuda-test" inkling2-dspark "$2"' - "$bin" "$model"
echo "measure: $(($(date +%s) - t1)) s"
