#!/bin/sh
# Card 129's contract + microbench job: the rebuilt h3 int8 tile's gates 1-4, then the tile-vs-cuBLASLt
# A/B on the four picture shapes (`h3-int8-bench`). The card's stop condition: if the tile does not beat
# cuBLASLt's int8 rate, stop before any picture run.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-129-contract.sh"
set -eu
# cuda-build.sh's contract leg execs tf-cuda-test, so unpack the toolchain and run its build line directly —
# gates 1-4, then this card's tile-vs-cuBLASLt A/B on the picture shapes.
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
[ -d "$staged" ] || { echo "missing $staged: run toolchain-stage.sh" >&2; exit 1; }
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc \
    -Doptimize=ReleaseFast --prefix /work/contract
BIN=/work/contract/bin/tf-cuda-test
"$BIN" kernel-contract "h3 int8"
"$BIN" h3-int8-bench | tee /work/out/h3-int8-bench.txt
