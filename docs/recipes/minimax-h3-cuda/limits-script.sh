#!/bin/sh
# Card 70 limits-only rerun (docs/kernel-contract.md:90) on TOP, through the Desk's GPU door:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/selftest-script.sh"
# The job has no network and no C++ compiler: the staged Zig 0.17.0 and Arm GNU 14.3.rel1 archives in
# /assets/drama3-pilot/cuda-build-toolchain (main's toolchain-stage.sh) build this commit in the job.
# Logs land in /work/out; the job exits 1 if a selftest, gate or limit fails.
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
mkdir -p "$out"
git -c safe.directory='*' rev-parse HEAD | tee "$out/sha"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$out/gpu"
# every served SM's records, read from the cubins built from this head, no other step
zig build install fatbins kernel-contract-fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121 \
    -Doptimize=ReleaseFast --prefix /work/sm121
failed=0
/work/sm121/bin/tf-cuda-test kernel-contract-limits /work/sm121/fatbin >"$out/limits.log" 2>&1 || failed=1
tail -n 5 "$out/limits.log"
exit "$failed"
