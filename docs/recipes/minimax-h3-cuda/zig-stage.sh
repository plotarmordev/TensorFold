#!/bin/sh
# Stages Zig 0.17.0 for cuda-build.sh, whose GPU job has no network: the official aarch64-linux archive (53 MB; the
# unpacked folder is 420 MB, too slow through `put`) lands in TOP's pilot folder as zig-0.17.0/, and the job checks its
# sha256 and unpacks it. The Desk runs it from any machine with network and the top-menu key, never on a Spark:
#   sh docs/recipes/minimax-h3-cuda/zig-stage.sh
set -eu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/zig-0.17.0"
curl -sSfL -o "$work/zig-0.17.0/zig-aarch64-linux-0.17.0.tar.xz" \
    https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
tar -C "$work" -c zig-0.17.0 | ssh top-menu "put top zig-0.17.0"
