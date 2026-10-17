#!/bin/sh
# build.sh SRC OUT [SMS]: fatbins for SMS (default 121), then the release-flag server, dev CLI and tf-cuda-test (contract
# controls built by nvcc) from SRC into OUT; records the commit, the commands and the sha256s.
set -eu
src=$(realpath "$1")
mkdir -p "$2"
out=$(realpath "$2")
sms=${3:-121}
zig="$HOME/opt/zig-aarch64-linux-0.17.0/zig"
nvcc=/usr/local/cuda/bin/nvcc
common="-Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline"
release="-Doptimize=fast -Dstrip=true -Dversion=1.0.12"
cd "$src"
{ git rev-parse HEAD; git status --short; } >"$out/commit"
{
    echo "zig 0.17.0, nvcc $($nvcc --version | tail -1)"
    echo "zig build fatbins $common -Dnvcc=<nvcc> -Dsm=$sms"
    echo "zig build native $common $release -Dfatbins=<out>/fatbin"
    echo "zig build install $common $release -Dsm=121 -Dnvcc=<nvcc>"
} >"$out/build"
"$zig" build fatbins $common -Dnvcc="$nvcc" -Dsm="$sms" --prefix "$out" -j2
"$zig" build native $common $release -Dfatbins="$out/fatbin" --prefix "$out" -j2
"$zig" build install $common $release -Dsm=121 -Dnvcc="$nvcc" --prefix "$out" -j2
cd "$out" && sha256sum native/bin/tensorfold-native bin/tensorfold bin/tf-cuda-test | tee "$out/sha256"
