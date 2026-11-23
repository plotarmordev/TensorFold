#!/bin/sh
# Builds the llama-cpp-b12628 folder that llama-cpp-job.sh reads from TOP's pilot folder (its GPU job has no network):
# the release's Ubuntu arm64 CUDA 13.5 binaries and their cudart bundle, and ggml-org/Inkling-2-Mini-GGUF's IQ4_NL (the
# mainline recipe's quant). b12628 was the newest release with those binaries when the job was written. Every file is
# checked against a pinned digest (the tarballs' are the release's published ones) before it is kept. Run on a machine
# with network; the Desk stages the folder on TOP:
#   sh docs/recipes/inkling-2-mini-cuda/llama-cpp-stage.sh [destination, default .]
set -eu
tag=b12628
stage=${1:-.}/llama-cpp-$tag
mkdir "$stage"
cd "$stage"
cat >SHA256SUMS <<EOF
2a6f7d5c65b511871cb6dbf7d610dd68767b18cfcc9a82218c432e1a838cb558  llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz
fe0cf0b1c16827065aa834dfcdd1391461bf915197f4a932eb868f8e1f2e3a53  cudart-llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz
cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4  Inkling-2-Mini-IQ4_NL.gguf
EOF
for asset in "llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz" "cudart-llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz"; do
    curl -sSfL -o "$asset" "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$asset"
done
curl -sSfL -O https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-IQ4_NL.gguf
shasum -a 256 -c SHA256SUMS
