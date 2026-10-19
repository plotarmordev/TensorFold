#!/bin/sh
# The release-flag server from SRC with OUT's fatbins (built by `zig build fatbins -Dsm=121`); its sha256 to stdout.
# Usage: native.sh SRC OUT
set -eu
src=$(realpath "$1") out=$(realpath "$2")
zig=$HOME/opt/zig-aarch64-linux-0.17.0/zig
cd "$src"
echo "$(git rev-parse HEAD) $(git status --short | tr '\n' ' ')"
"$zig" build native -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline -Doptimize=fast -Dstrip=true -Dversion=1.1.0 \
    -Dfatbins="$out/fatbin" --prefix "$out" -j1
cd "$out" && sha256sum native/bin/tensorfold-native
