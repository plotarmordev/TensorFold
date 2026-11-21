#!/bin/sh
# Card 147's stop-rule microbench: GB10's native block-scaled NVFP4 products (e2m1 at one e4m3 scale every
# 16 values, cuBLASLt VEC16_UE4M3, fp32 sums, bf16 out) against the int8 path on the four picture shapes.
# Pure cuBLASLt: tf-cuda-test builds without nvcc or the gcc toolchain. The dtype and scale-mode ints come
# from the toolkit's own headers, so a wrong enum value cannot fake a result.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-147-fp4.sh"
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
[ -d "$staged" ] || { echo "missing $staged: run toolchain-stage.sh" >&2; exit 1; }
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
EOF
tar -C /work -xJf "$staged/$zig.tar.xz"
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
inc=/usr/local/cuda/include
dtype=$(grep -oE 'CUDA_R_4F_E2M1[[:space:]]*=[[:space:]]*[0-9]+' "$inc/library_types.h" | grep -oE '[0-9]+$')
mode=$(grep -oE 'CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3[[:space:]]*=[[:space:]]*[0-9]+' "$inc/cublasLt.h" | grep -oE '[0-9]+$')
echo "header CUDA_R_4F_E2M1=$dtype CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3=$mode"
[ -n "$dtype" ] && [ -n "$mode" ] || { echo "cannot read the header enums" >&2; exit 1; }
zig build install -Doptimize=ReleaseFast --prefix /work/bench
/work/bench/bin/tf-cuda-test h3-fp4-bench "$dtype" "$mode" | tee /work/out/h3-fp4-bench.txt
