#!/bin/sh
# Builds this commit's CUDA kernel images and tf-cuda-test on TOP's Spark and audits the images' limits, through the
# Desk's GPU door (never `bottom shell` or `middle shell`: those Sparks serve DeepSeek):
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/cuda-build.sh"
# The job has no network and no C++ compiler: toolchain-stage.sh puts the archives of Zig 0.17.0 and of Arm's GNU
# toolchain (gcc 14, nvcc's host compiler) in the pilot folder, and the job unpacks them. In /work/out:
# - tensorfold-fatbins-SHA: the sm_121 images and SHA256SUMS, for -Dfatbins (../../receipts/drama3-pruned-main/run-script.sh)
# - tensorfold-cuda-SHA: tf-cuda-test and the three H3 picture tools, with their images built in, plus fatbin/ for the five release SMs
# - limits.log: kernel-contract-limits over those images; the job exits with its status
# H3_BUILD=sm121 stops once the three picture tools are in tensorfold-cuda-SHA: no release SMs, tf-cuda-test or
# audit, which put bench-script.sh past the Desk's 20 minutes.
# H3_BUILD=limits builds the five SMs' images and kernel-contract-limits alone, no CLI or tf-cuda-test: the audit
# as its own job, never inside a timed bench.
# H3_BUILD=contract builds tf-cuda-test for sm_121 alone and runs gates 1-4 on the kernels H3_CONTRACT's
# words name (`kernel-contract WORD...`, "h3 int8" unset): contract-job.sh asks for the decoder suite.
# Later TOP jobs read either folder from the pilot folder once the Desk returns it there:
#   ssh top-menu "job fetch ID" | tar -x -C DIR
#   tar -C DIR/out -c tensorfold-fatbins-SHA | ssh top-menu "put top tensorfold-fatbins-SHA"
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
sha=$(git -c safe.directory='*' rev-parse HEAD)
out=/work/out
nvcc=/usr/local/cuda/bin/nvcc
[ -d "$staged" ] || { echo "missing $staged: run toolchain-stage.sh" >&2; exit 1; }
# the official archives' sha256, from ziglang.org's release index and Arm's .sha256asc
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
# H3_BUILD=limits: the five SMs' images and kernel-contract-limits alone, no picture tools, CLI or tf-cuda-test —
# the audit as its own job, never inside a timed bench
cuda=$out/tensorfold-cuda-$sha
mkdir -p "$cuda"
if [ "${H3_BUILD:-}" = limits ]; then
    zig build fatbins kernel-contract-fatbins tf-cuda-limits -Dsm=80,86,89,120,121 -Dnvcc=$nvcc -Doptimize=ReleaseFast \
        --prefix /work/release
    cp -r /work/release/bin/tf-cuda-limits /work/release/fatbin "$cuda"
    status=0
    "$cuda/tf-cuda-limits" "$cuda/fatbin" h3 h3_image_decoder h3_image_prompt h3_image_modulation contract_controls \
        >"$out/limits.log" 2>&1 || status=$?
    tail -n 3 "$out/limits.log"
    echo "kernel-contract-limits exit $status"
    exit "$status"
fi
if [ "${H3_BUILD:-}" = contract ]; then
    zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix /work/contract
    exec /work/contract/bin/tf-cuda-test kernel-contract ${H3_CONTRACT:-h3 int8}
fi
# -Dfatbins takes images holding the one SASS profile, so sm_121 is built apart from the audit's five SMs
zig build fatbins tf-h3-image-prompt-cuda tf-h3-image-cuda tf-h3-image-decoder-cuda \
    -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix /work/sm121
fatbins=$out/tensorfold-fatbins-$sha
mkdir -p "$fatbins"
cp /work/sm121/fatbin/*.fatbin "$fatbins"
(cd "$fatbins" && sha256sum *.fatbin >SHA256SUMS)
cp /work/sm121/bin/tf-h3-image-prompt-cuda /work/sm121/bin/tf-h3-image-cuda /work/sm121/bin/tf-h3-image-decoder-cuda "$cuda"
[ "${H3_BUILD:-}" != sm121 ] || exit 0
zig build install fatbins kernel-contract-fatbins -Dsm=80,86,89,120,121 -Dnvcc=$nvcc -Doptimize=ReleaseFast \
    --prefix /work/release
cp -r /work/release/bin/tf-cuda-test /work/release/fatbin "$cuda"
status=0
"$cuda/tf-cuda-test" kernel-contract-limits "$cuda/fatbin" >"$out/limits.log" 2>&1 || status=$?
tail -n 3 "$out/limits.log"
echo "kernel-contract-limits exit $status"
exit "$status"
