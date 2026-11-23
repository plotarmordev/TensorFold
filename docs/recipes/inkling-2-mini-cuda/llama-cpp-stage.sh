#!/bin/sh
# Stages llama.cpp b12628 for llama-cpp-job.sh, whose GPU job has no network: the release's Ubuntu arm64 CUDA 13.5
# binaries and their cudart bundle, and ggml-org/Inkling-2-Mini-GGUF's IQ4_NL (the mainline recipe's quant), in TOP's
# pilot folder as llama-cpp-b12628. b12628 is the first release with #32704's loop-slot NVFP4 scale fix. Run from a
# machine with network and the top-menu key, never on a Spark:
#   sh docs/recipes/inkling-2-mini-cuda/llama-cpp-stage.sh
set -eu
tag=b12628
release=https://github.com/ggml-org/llama.cpp/releases/download/$tag
gguf=Inkling-2-Mini-IQ4_NL.gguf
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
stage=$work/llama-cpp-$tag
mkdir "$stage"
for asset in "llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz" "cudart-llama-$tag-bin-ubuntu-cuda-13.5-arm64.tar.gz"; do
    curl -sSfL -o "$stage/$asset" "$release/$asset"
done
curl -sSfL -o "$stage/$gguf" "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/$gguf"
echo "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4  $stage/$gguf" | shasum -a 256 -c
(cd "$stage" && shasum -a 256 *.tar.gz *.gguf >SHA256SUMS)
tar -C "$work" -c "llama-cpp-$tag" | ssh top-menu "put top llama-cpp-$tag"
