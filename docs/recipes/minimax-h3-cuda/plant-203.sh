#!/bin/sh
# PR 3733's ask 5, the rest of the GB10 kernel contract: TOP job top-a4a49aacbcd1 stopped at 45 min in window_rows, so
# this job resumes gates 1-4 at its first kernel (window_rows through family-set), then the back and after suites, the
# live limits and the SUMMARY, still without moss's tf_md_conv3x3 float64 reference (the Lead's call). Cap: 60 min.
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
out=/work/out
nvcc=/usr/local/cuda/bin/nvcc
[ -d "$staged" ] || { echo "missing $staged: run toolchain-stage.sh" >&2; exit 1; }
# the official archives' sha256, from ziglang.org's release index and Arm's .sha256asc
(cd "$staged" && sha256sum -c --quiet) <<SUMS
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
SUMS
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
mkdir -p $out
git -c safe.directory='*' rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
b=/work/b
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix $b
c=0
timeout 3000 $b/bin/tf-cuda-test kernel-contract @window_rows -tf_md_conv3x3 >$out/contract.log 2>&1 || c=$?
echo "== contract from window_rows without tf_md_conv3x3: exit $c (124: stopped at 50 min), $(wc -l <$out/contract.log) lines"
cat $out/contract.log
exit "$c"
