#!/bin/sh
# bench-script.sh's warm pictures under the CUPTI preload, so TensorFold's own kernels get the same trace
# mmh3-server-job.sh takes of mmh3's: one TOP window, both engines' kernel budgets.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-trace-job.sh"
# The preload writes a cupti-trace.csv.<pid> a process in /work/out/h3-image-bench-trace; the three resident
# servers' files (prompt, transformer, decoder) land beside bench-script.sh's own logs and SUMMARY. The
# transformer server's segment for the last warm picture is our side of the study's kernel table.
set -eu
out=/work/out/h3-image-bench-trace
gccdir=/work/arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
if [ ! -f "$gccdir/bin/aarch64-none-linux-gnu-gcc" ]; then
    tar -C /work -xJf /assets/drama3-pilot/cuda-build-toolchain/arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu.tar.xz
fi
"$gccdir/bin/aarch64-none-linux-gnu-gcc" -shared -fPIC -o /work/cupti-trace.so \
    docs/recipes/minimax-h3-cuda/cupti-trace.c \
    -I/usr/local/cuda/targets/sbsa-linux/include -L/usr/local/cuda/targets/sbsa-linux/lib -lcupti \
    && echo "cupti-trace: compiled" || { echo "cupti-trace: COMPILE FAILED"; exit 1; }
rm -rf "$out"
mkdir -p "$out"
H3_DECODER=tc CUPTI_TRACE_OUT="$out/cupti-trace.csv" LD_PRELOAD=/work/cupti-trace.so \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh
# bench-script.sh's own $out is /work/out/h3-image-bench; fold its logs beside the traces
cp -r /work/out/h3-image-bench/. "$out/"
echo "bench-trace-job done"
