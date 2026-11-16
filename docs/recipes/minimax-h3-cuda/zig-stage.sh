#!/bin/sh
# Stages Zig 0.17.0 for cuda-build.sh, whose GPU job has no network: the official aarch64-linux build, checked against
# its sha256 (docs/recipes/adding-a-zig-family.md), lands in TOP's pilot folder as zig-aarch64-linux-0.17.0. The Desk
# runs it from any machine with network and the top-menu key, never on a Spark:
#   sh docs/recipes/minimax-h3-cuda/zig-stage.sh
set -eu
name=zig-aarch64-linux-0.17.0
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -sSfL -o "$work/$name.tar.xz" "https://ziglang.org/download/0.17.0/$name.tar.xz"
echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $work/$name.tar.xz" | shasum -a 256 -c -
tar -C "$work" -xJf "$work/$name.tar.xz"
tar -C "$work" -c "$name" | ssh top-menu "put top $name"
