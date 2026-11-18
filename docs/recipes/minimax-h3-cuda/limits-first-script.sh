#!/bin/sh
# Card 70 on TOP's GB10, through the Desk's GPU door (not for merge):
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/limits-first-script.sh"
# The limits (docs/kernel-contract.md:90) run first, so a later step that dies cannot take them; the gates and
# selftests follow. Builds run 4 nvcc at a time: the default, one per core, was SIGKILLed inside the job's 96 GB.
# The job has no network and no C++ compiler: the staged Zig 0.17.0 and Arm GNU 14.3.rel1 archives in
# /assets/drama3-pilot/cuda-build-toolchain (main's toolchain-stage.sh) build this commit in the job.
# Logs land in /work/out, one per step; the job exits 1 if a limit, gate or selftest fails.
set -eu
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
out=/work/out
bin=/work/sm/bin
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
build() { zig build -j4 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121 -Doptimize=ReleaseFast --prefix /work/sm "$@"; }
mkdir -p "$out"
git -c safe.directory='*' rev-parse HEAD | tee "$out/sha"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$out/gpu"
failed=0
# every served SM's records, read from the cubins built from this head, no GPU
build install fatbins kernel-contract-fatbins
rc=0
"$bin/tf-cuda-test" kernel-contract-limits /work/sm/fatbin >"$out/limits.log" 2>&1 || rc=$?
echo "limits exit $rc" | tee "$out/limits.exit"
[ "$rc" -eq 0 ] || failed=1
tail -n 5 "$out/limits.log"
# gates 1-4 on every registered kernel and gate 5 live, one SUMMARY line
"$bin/tf-cuda-test" kernel-contract >"$out/kernel-contract.log" 2>&1 || failed=1
build tf-h3-image-decoder-cuda tf-h3-image-prompt-cuda
"$bin/tf-h3-image-decoder-cuda" selftest >"$out/decoder-selftest.log" 2>&1 || failed=1
"$bin/tf-h3-image-prompt-cuda" selftest "$out/prompt-selftest" >"$out/prompt-selftest.log" 2>&1 || failed=1
tail -n 5 "$out/kernel-contract.log" "$out/decoder-selftest.log" "$out/prompt-selftest.log"
exit "$failed"
