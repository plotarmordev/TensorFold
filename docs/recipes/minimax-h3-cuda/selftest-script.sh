#!/bin/sh
# MiniMax-H3-Image's CUDA prompt and decoder selftests on TOP's Spark, through the Desk's GPU door:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/selftest-script.sh"
# The job has no network and no C++ compiler: the staged Zig 0.17.0 and Arm GNU 14.3.rel1 archives in
# /assets/drama3-pilot/cuda-build-toolchain (main's toolchain-stage.sh) build this commit's two tools in the job.
# Each selftest's log lands in /work/out; the job exits 1 if either fails.
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
out=/work/out
[ -d "$staged" ] || { echo "missing $staged" >&2; exit 1; }
# the official archives' sha256, from ziglang.org's release index and Arm's .sha256asc
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
zig build tf-h3-image-decoder-cuda tf-h3-image-prompt-cuda -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
    -Doptimize=ReleaseFast --prefix /work/sm121
mkdir -p "$out"
git -c safe.directory='*' rev-parse HEAD | tee "$out/sha"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$out/gpu"
failed=0
/work/sm121/bin/tf-h3-image-decoder-cuda selftest >"$out/decoder-selftest.log" 2>&1 || failed=1
/work/sm121/bin/tf-h3-image-prompt-cuda selftest /work/out/prompt-selftest >"$out/prompt-selftest.log" 2>&1 || failed=1
tail -n 30 "$out/decoder-selftest.log" "$out/prompt-selftest.log"
exit "$failed"
