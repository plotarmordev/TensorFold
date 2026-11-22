#!/bin/sh
# PR 3733 respond check on the GB10: kernel-contract h3 (gates and the planted controls), both CUDA selftests, limits.
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
b=/work/b
zig build install fatbins kernel-contract-fatbins tf-h3-image-prompt-cuda tf-h3-image-decoder-cuda -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix $b
status=0
$b/bin/tf-cuda-test kernel-contract h3 >$out/contract.log 2>&1 || status=$?
grep -E "CONTROL|^FAIL|SUMMARY|panic" $out/contract.log | tail -n 80
echo "contract exit $status"
for t in decoder prompt; do
    s=0
    $b/bin/tf-h3-image-$t-cuda selftest >$out/$t-selftest.txt 2>&1 || s=$?
    grep -E "FAIL|selftest" $out/$t-selftest.txt | tail -n 4
    echo "$t selftest exit $s"
    [ "$s" = 0 ] || status=$s
done
zig build fatbins kernel-contract-fatbins -Dsm=80,86,89,120,121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix /work/release
l=0
$b/bin/tf-cuda-test kernel-contract-limits /work/release/fatbin >$out/limits.log 2>&1 || l=$?
grep -E "h3p_|h3d_" $out/limits.log | head -n 20
tail -n 3 $out/limits.log
echo "limits exit $l"
exit "$status"
