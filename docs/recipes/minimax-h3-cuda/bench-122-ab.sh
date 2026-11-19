#!/bin/sh
# Card 122's same-job A/B: bench this commit's h3 image tools, then the same tools with the card's files
# reverted to the merge base (main before the card), in the one job — the "keep only if faster" gate.
# The reverted decoder is main's fp32 default, so its phase passes --tc explicitly. Results land in
# /work/out/h3-image-bench (after/) and h3-image-bench-base (before/).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-122-ab.sh"
set -eu
base=b044c5ff3acd894dc04f5bb4651d6adb1d2d0958
files="zig/src/families/h3/cuda.zig zig/src/families/h3/cuda_gemm.zig zig/src/families/h3/cuda_int8.zig \
zig/src/families/h3/cuda_rows.zig zig/src/families/h3/image_decoder_cuda.zig zig/src/families/h3/picture.zig \
zig/tests/h3_image_cuda.zig zig/tests/h3_image_decoder_cuda.zig zig/build/h3_image_cuda.zig"
# shellcheck disable=SC2086 # the file list must word-split
saved=$(mktemp -d)
# shellcheck disable=SC2086
for f in $files; do mkdir -p "$saved/$(dirname "$f")"; cp "$f" "$saved/$f"; done
status=0
echo "=== after: this commit ==="
H3_BENCH_OUT=/work/out/h3-image-bench sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
# shellcheck disable=SC2086
git -c safe.directory='*' checkout "$base" -- $files
echo "=== before: $base files on this commit, --tc ==="
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_DECODER=tc H3_BENCH_SKIP_COMPARE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
# shellcheck disable=SC2086
for f in $files; do cp "$saved/$f" "$f"; done
exit "$status"
