#!/bin/sh
# PR 3733's review round on the GB10: which H3 clip checkpoints and case files this host holds (ask 4, kept in
# checkpoints.txt), and the full kernel contract (ask 5) within the Desk's 30 min, its log kept as far as it gets.
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
timeout 300 find / -xdev \( -iname '*fasth3*' -o -iname '*h3*case*' -o -iname 'case.safetensors' -o -iname '*minimax*h3*' \) \
    -not -path '/proc/*' -not -path '/work/*' 2>/dev/null >$out/checkpoints.txt || true
echo "$(wc -l <$out/checkpoints.txt) paths"; cat $out/checkpoints.txt
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
b=/work/b
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix $b
c=0
timeout 1200 $b/bin/tf-cuda-test kernel-contract >$out/contract.log 2>&1 || c=$?
echo "== full contract exit $c (124: stopped at 20 min), $(wc -l <$out/contract.log) lines; last suite reached:"
tail -n 3 $out/contract.log
grep -cE "^PASS" $out/contract.log || true
echo "== every FAIL, CONTROL and KNOWN line"
grep -E "^(FAIL|CONTROL|KNOWN)" $out/contract.log || true
grep -E "SUMMARY|^FAIL kernel-contract" $out/contract.log || true
