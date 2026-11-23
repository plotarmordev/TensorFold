#!/bin/sh
# Strata's own Inkling 2 Mini engine on TOP's GB10, for card 209's measured gap:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/strata-inkling/strata-job.sh
# The job has no network: the Desk staged strata-inkling 0.1.66 (linux-aarch64, CUDA 13; its cuBLAS, cudart and
# libgomp ride along) and the SQ-10 pack in the pilot folder strata-inkling-pack/. The engine runs as shipped, nothing is
# installed: its built-in `--bench` (DSpark drafting as the engine defaults it) at 1, 4 and 64 streams, 256 generated
# tokens each, then `--selfcheck`. Logs land in /work/out/strata-inkling; the `[bench]` lines go to stdout.
# The binary needs glibc 2.38 and GLIBCXX 3.4.32, newer than the job image's, so it runs under the loader, glibc and
# libstdc++ of Arm's GNU 14.3 toolchain that the Desk staged for cuda-build.sh, unpacked in /work.
set -eu
dir=/assets/drama3-pilot/strata-inkling-pack
pack=$dir/Inkling-2-Mini-SQ-10.strata
engine=$(find "$dir" -name strata-inkling -type f | head -1)
out=/work/out/strata-inkling
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
toolchain=/assets/drama3-pilot/cuda-build-toolchain/$gcc.tar.xz
[ -f "$pack" ] && [ -n "$engine" ] || { echo "missing pack or engine under $dir" >&2; exit 1; }
echo "ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934  $pack" | sha256sum -c --quiet
engine_dir=$(dirname "$engine")
{
    echo "e96a5ce6c49d45d6355e0fccd881e094d36b80ea8722e0d8307388ebd1746c6d  $engine"
    echo "930418840b747413584867cb7167dd5911dcf4184135589e0334527e9abbac9f  $engine_dir/libcublasLt.so.13"
    echo "4e346a7b60a01552bd9bf4bbc6c862e11eb851559c3da3572293345de6aaed75  $engine_dir/libcublas.so.13"
    echo "7bdba2b5b08cbdc85203c41cc94598adedb1bcfea7cb574ca693ac73599e4e63  $engine_dir/libcudart.so.13"
    echo "33c8bcb7d33228fe5101bfeb201bcaff4563d4ecf09497229b9deba522ebaf3d  $engine_dir/libgomp.so.1"
} | sha256sum -c --quiet
echo "7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $toolchain" | sha256sum -c --quiet
tar -C /work -xJf "$toolchain"
loader=$(find "/work/$gcc" -name ld-linux-aarch64.so.1 | head -1)
libc=$(dirname "$(find "/work/$gcc" -name libc.so.6 | head -1)")
libstdcxx=$(dirname "$(find "/work/$gcc" -name libstdc++.so.6 | head -1)")
libs=$engine_dir:$libc:$libstdcxx:/usr/lib/aarch64-linux-gnu
strata() { "$loader" --library-path "$libs" "$engine" "$@"; }
mkdir -p "$out"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
strata --version
status=0
for streams in 1 4 64; do
    strata --bench --native "$pack" --parallel "$streams" --gen 256 \
        >"$out/bench-$streams.log" 2>&1 || status=$?
    grep '^\[bench\]' "$out/bench-$streams.log" || { tail -20 "$out/bench-$streams.log"; status=1; }
done
strata --selfcheck --native "$pack" >"$out/selfcheck.log" 2>&1 || status=$?
tail -5 "$out/selfcheck.log"
exit "$status"
