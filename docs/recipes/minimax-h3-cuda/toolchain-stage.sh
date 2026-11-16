#!/bin/sh
# Stages the toolchain of cuda-build.sh, whose GPU job has no network and no C++ compiler: the official archives of
# Zig 0.17.0 (the build) and Arm's GNU toolchain 14.3.rel1 (nvcc's host compiler, gcc 14 with its own glibc sysroot;
# Zig's clang and libc++ fail CUDA 13.0's front end), for aarch64 Linux, land in TOP's pilot folder as
# cuda-build-toolchain/; the job checks their sha256 and unpacks them. The Desk runs it from any machine with network
# and the top-menu key, never on a Spark:
#   sh docs/recipes/minimax-h3-cuda/toolchain-stage.sh
set -eu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
dir=$work/cuda-build-toolchain
mkdir "$dir"
curl -sSfL -o "$dir/zig-aarch64-linux-0.17.0.tar.xz" https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
curl -sSfL -o "$dir/arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu.tar.xz" \
    https://developer.arm.com/-/media/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu.tar.xz
tar -C "$work" -c cuda-build-toolchain | ssh top-menu "put top cuda-build-toolchain"
