#!/bin/sh
# Builds the llama-cpp-b12628 folder that llama-cpp-job.sh reads from TOP's pilot folder (its GPU job has no network):
# the release's Ubuntu arm64 CUDA 13.5 binaries and their cudart bundle, and ggml-org/Inkling-2-Mini-GGUF's IQ4_NL (the
# mainline recipe's quant). b12628 is the release the GB10 receipt measured; the job checks every file against the
# digests it pins before running anything. Run on a machine with network; the Desk stages the folder on TOP:
#   sh docs/recipes/inkling-2-mini-cuda/llama-cpp-stage.sh [destination, default .]
set -eu
tag=b12628
stage=${1:-.}/llama-cpp-$tag
mkdir "$stage"
cd "$stage"
for asset in "llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz" "cudart-llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz"; do
    curl -sSfL -o "$asset" "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$asset"
done
curl -sSfL -O https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-IQ4_NL.gguf
