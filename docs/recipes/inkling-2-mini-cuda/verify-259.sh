#!/bin/sh
# Card 259's GB10 checks on TOP, through `swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/verify-259.sh`,
# on 188's toolchain and checkpoint (../minimax-h3-cuda/toolchain-stage.sh). Each step runs under `timeout 600`:
# - limits: the Inkling 2 images for the five release SMs against their records; its RECORD lines are the new launches' usage
# - contract: tf-cuda-test kernel-contract over the Inkling 2 kernels, gateParallel and tf_ink2_rowdot3 among them
# - serving: tf-cuda-test inkling2-serving, main's host-reference, resume, admission, concurrent and rollback checks
# Logs land in /work/out/inkling2-259-verify.
set -eu
pilot=/assets/drama3-pilot
out=/work/out/inkling2-259-verify
staged=$pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
model=
for dir in "$pilot/Inkling-2-Mini" "$pilot/models/Inkling-2-Mini"; do
    [ -f "$dir/config.json" ] && model=$dir && break
done
[ -n "$model" ] || { echo "no Inkling-2-Mini/config.json in $pilot or $pilot/models" >&2; exit 1; }
[ -d "$staged" ] || { echo "missing $staged: run ../minimax-h3-cuda/toolchain-stage.sh" >&2; exit 1; }
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
rm -rf "$out"
mkdir -p "$out"
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
nvcc=/usr/local/cuda/bin/nvcc
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
step() { # step NAME COMMAND...: one check under its own 10-minute cap, its log in $out/NAME.log
    name=$1
    shift
    start=$(date +%s)
    status=0
    timeout --kill-after=20 600 "$@" >"$out/$name.log" 2>&1 || status=$?
    grep -E '^(RECORD|RESULT|PASS|FAIL)' "$out/$name.log" | grep -E 'ink2|inkling2|Inkling|Mini|RESULT|FAIL' || :
    tail -n 3 "$out/$name.log"
    echo "measure $name: $(($(date +%s) - start)) s, exit $status"
    return "$status"
}
zig build fatbins tf-cuda-limits -Dsm=80,86,89,120,121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix /work/release
zig build install fatbins kernel-contract-fatbins -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseFast --prefix /work/sm121
python3 -c 'import random; r = random.Random(4387); print(" ".join(str(r.randrange(1000, 199000)) for _ in range(4200)))' >"$out/ids.txt"
status=0
step limits /work/release/bin/tf-cuda-limits /work/release/fatbin inkling2_ops inkling2_loop_math || status=1
step contract /work/sm121/bin/tf-cuda-test kernel-contract inkling2 || status=1
step serving /work/sm121/bin/tf-cuda-test inkling2-serving "$model" "$out/ids.txt" || status=1
exit "$status"
