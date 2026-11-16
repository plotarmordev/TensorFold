#!/bin/sh
# The H3 image launch records against nvcc's five served SM images, no GPU; usage in ../minimax-h3-cuda.md.
set -eu
nvcc=${1:-/usr/local/cuda/bin/nvcc}
root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$root"
zig build tf-cuda-limits fatbins kernel-contract-fatbins -Dnvcc="$nvcc" -Dsm=80,86,89,120,121 -Doptimize=ReleaseFast
zig-out/bin/tf-cuda-limits zig-out/fatbin h3 h3_image_decoder h3_image_prompt h3_image_modulation contract_controls
