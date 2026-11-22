#!/bin/sh
# PR 3733 asks 3-5 on the GB10: the full kernel contract (gates and every control), both CUDA selftests,
# the five-SM limits audit, and which H3 clip checkpoints this host holds (ask 4).
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
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
mkdir -p $out
git -c safe.directory='*' rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "== ask 4: FastH3 / H3 clip checkpoints and case files on this host"
find / -xdev \( -iname '*fasth3*' -o -iname '*.h3case.safetensors' -o -iname 'case.safetensors' \) 2>/dev/null | head -n 20 || true
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
b=/work/b
zig build install fatbins kernel-contract-fatbins tf-h3-image-prompt-cuda tf-h3-image-decoder-cuda -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix $b
status=0
for t in decoder prompt; do
    s=0
    $b/bin/tf-h3-image-$t-cuda selftest >$out/$t-selftest.txt 2>&1 || s=$?
    tail -n 4 $out/$t-selftest.txt
    echo "== $t selftest exit $s"
    [ "$s" = 0 ] || status=$s
done
c=0
$b/bin/tf-cuda-test kernel-contract >$out/contract.log 2>&1 || c=$?
echo "== ask 3: every H3 image control and the gate line it failed"
grep -F " control: first block only" $out/contract.log
echo "== H3 gate failures, controls aside"
grep -E "^FAIL .*h3" $out/contract.log | grep -vF " control: first block only" || true
grep -E "^(FAIL|CONTROL MISSED|CONTROL ERROR|KNOWN)" $out/contract.log | grep -vF " control: first block only" | tail -n 60 || true
grep -E "SUMMARY|^FAIL kernel-contract" $out/contract.log
echo "== contract exit $c"
zig build fatbins kernel-contract-fatbins -Dsm=80,86,89,120,121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix /work/release
l=0
$b/bin/tf-cuda-test kernel-contract-limits /work/release/fatbin >$out/limits.log 2>&1 || l=$?
echo "== limits: H3 records"
grep -E "h3d_|h3p_|h3_" $out/limits.log | sort | uniq -c | sort -rn | head -n 40 || true
echo "== limits: every non-current line"
grep -vE "record current|^INFO" $out/limits.log || true
echo "== limits exit $l"
exit "$status"
