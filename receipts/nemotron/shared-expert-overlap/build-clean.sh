#!/bin/sh
# build-clean.sh SRC OUT [native]: fatbins, install and optionally native from SRC, each command logged into OUT.
set -eu
Z=zig  # 0.17.0
src=$1 out=$(realpath -m "$2")
mkdir -p "$out"
cd "$src"
git rev-parse HEAD > "$out/commit.txt"
git status --short >> "$out/commit.txt"
run() { echo "zig build $*" | sed "s|$out|<out>|g" >> "$out/commands.txt"; $Z build "$@" > "$out/$1.log" 2>&1; }
run fatbins -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix "$out" -j2
common="-Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline -Doptimize=fast -Dstrip=true -Dversion=1.0.12 -Dfatbins=$out/fatbin --prefix $out -j2"
run install -Dsm=121 $common
if [ "${3:-}" = native ]; then run native $common; fi
