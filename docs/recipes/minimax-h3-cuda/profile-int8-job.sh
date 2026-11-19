#!/bin/sh
# Card 129: profile the reverted h3_gemm_int8 (this commit is cdf02b8b4 + this script) on TOP's GB10 —
# achieved TOPS, occupancy and smem/register limits for the recipe's loss analysis. Falls back to the
# contract's gate run if ncu is absent from the image.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/profile-int8-job.sh"
set -eu
export H3_BUILD=contract
sh docs/recipes/minimax-h3-cuda/cuda-build.sh
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
