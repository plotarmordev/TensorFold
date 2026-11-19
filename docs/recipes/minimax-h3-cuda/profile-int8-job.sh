#!/bin/sh
# Card 129: profile the reverted h3_gemm_int8 (this commit is cdf02b8b4 + this script) on TOP's GB10 —
# achieved TOPS, occupancy and smem/register limits for the recipe's loss analysis. Falls back to the
# contract's gate run if ncu is absent from the image. cuda-build.sh's contract leg execs tf-cuda-test,
# so the toolchain unpack and the build line are inlined here.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/profile-int8-job.sh"
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
[ -d "$staged" ] || { echo "missing $staged: run toolchain-stage.sh" >&2; exit 1; }
(cd "$staged" && sha256sum -c --quiet) <<SHAEOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
SHAEOF
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc \
    -Doptimize=ReleaseFast --prefix /work/contract
BIN=/work/contract/bin/tf-cuda-test
NCU=""
for c in ncu /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then NCU="$c"; break; fi
done
echo "ncu: ${NCU:-MISSING}"
if [ -n "$NCU" ]; then
    # contract run under the profiler, limited to the gemm kernel's launches
    "$NCU" --kernel-name-base demangled -k h3_gemm_int8 \
        --set full --section SchedulerStats --section Occupancy \
        --section SpeedOfLight --section WarpStateStats \
        -o /work/out/h3-gemm-int8-ncu \
        "$BIN" kernel-contract "h3 int8" || true
    "$NCU" --import /work/out/h3-gemm-int8-ncu.ncu-rep --page details || true
else
    # no profiler in the image: the gates' own timing is the only device evidence
    "$BIN" kernel-contract "h3 int8"
    # register/smem/readout from the embedded fatbin instead of a profile
    cuobjdump -res-usage /work/contract/fatbin/h3.fatbin 2>/dev/null | grep -A12 "h3_gemm_int8" || true
fi
