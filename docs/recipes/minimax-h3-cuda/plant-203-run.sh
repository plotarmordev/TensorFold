#!/bin/sh
# PR 3733's ask 5, the rest of the GB10 kernel contract, one part per TOP job: plant-203-run.sh PART.
# gN: gates 1-4 from window_rows on, shard %N/6 (without tf_md_conv3x3's reference, the Lead's call);
# back, after, limits: one part of the full run's tail. Build and measure are separate phases; the measure stops at 600 s.
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
nvcc=/usr/local/cuda/bin/nvcc
[ -d "$staged" ] || { echo "missing $staged: run toolchain-stage.sh" >&2; exit 1; }
# the official archives' sha256, from ziglang.org's release index and Arm's .sha256asc
(cd "$staged" && sha256sum -c --quiet) <<SUMS
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
SUMS
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
git -c safe.directory='*' rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
part=${1:?part: g0..g5, back, after or limits}
case $part in
g[0-5]) cmd="kernel-contract @window_rows %${part#g}/6 -tf_md_conv3x3" ;;
back|after|limits) cmd="kernel-contract-suites $part" ;;
*) echo "unknown part $part" >&2; exit 2 ;;
esac
b=/work/b
echo "== build"
s=$(date +%s)
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix $b
echo "== build: $(( $(date +%s) - s )) s"
echo "== measure: tf-cuda-test $cmd"
s=$(date +%s)
c=0
timeout 600 $b/bin/tf-cuda-test $cmd >/work/measure.log 2>&1 || c=$?
# the door keeps a log's last 200 lines: PASS lines are only counted, every other line is printed
echo "== measure: exit $c (124: stopped at 600 s) in $(( $(date +%s) - s )) s, $(grep -c '^PASS' /work/measure.log) PASS lines"
grep -v '^PASS' /work/measure.log || true
exit "$c"
