#!/bin/sh
# Inkling 2 Mini's DSpark draft head (mtp.safetensors) on TOP's GB10 against its float64 host reference:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/dspark.sh
# The head's kernels under the contract (FP64 references, limits, planted controls), then seven device-fed greedy
# drafts after 1,100 seeded context rows: every step's logits within 2% RMS and 0.25 a logit of the reference
# teacher-forced on the device's picks, each pick within 0.25 of the reference's best, then the planted context-blind
# reference, which must go red.
set -eu
. docs/recipes/inkling-2-mini-cuda/toolchain.sh
t0=$(date +%s)
zig build install -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/sm121
# The head's per-SM compiled usage for its launch records, at the image's flags.
for sm in 80 86 89 120 121; do
  /usr/local/cuda/bin/nvcc -cubin -o /dev/null --resource-usage -arch=sm_$sm -O3 --fmad=false --ftz=false --prec-div=true \
    --prec-sqrt=true -std=c++20 -Izig/kernels/cuda zig/kernels/cuda/inkling2/dspark.cu 2>&1 | sed -n "s/^ptxas info *: \(Used\|Function properties\|Compiling entry\)/sm_$sm &/p"
done
t1=$(date +%s)
echo "build: $((t1 - t0)) s"
timeout 600 sh -c '"$1/tf-cuda-test" kernel-contract "inkling2 dspark" && "$1/tf-cuda-test" inkling2-dspark "$2" && INKLING2_REF_PLANT=1 "$1/tf-cuda-test" inkling2-dspark "$2"' - "$bin" "$model"
echo "measure: $(($(date +%s) - t1)) s"
