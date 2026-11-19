#!/bin/sh
# Card 130's same-job A/B: bench this commit's h3 image tools, then the same tools with the card's
# files reverted to the merge base (main before the card), in the one job — the "keep only if faster"
# gate. Both phases run the default --tc decoder. Results land in /work/out/h3-image-bench (after/)
# and h3-image-bench-base (before/).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-130-ab.sh"
set -eu
base=60dd5ac5ac6ad95a93a431e090199ad07ddcf50b
files="zig/src/families/h3/image_decoder_cuda.zig zig/kernels/cuda/h3_image_decoder.cu \
zig/src/cuda/vit.zig zig/tests/h3_image_decoder_cuda.zig \
zig/tests/cuda/contract_h3_image_decoder.zig zig/tests/cuda/contract_h3_image_ref.zig \
zig/src/core/contract/h3_image_cuda_records.zig"
# shellcheck disable=SC2086 # the file list must word-split
saved=$(mktemp -d)
# shellcheck disable=SC2086
for f in $files; do mkdir -p "$saved/$(dirname "$f")"; cp "$f" "$saved/$f"; done
status=0
echo "=== after: this commit ==="
H3_BENCH_OUT=/work/out/h3-image-bench sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
# shellcheck disable=SC2086
git -c safe.directory='*' checkout "$base" -- $files
echo "=== before: $base files on this commit ==="
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_BENCH_SKIP_COMPARE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
# shellcheck disable=SC2086
for f in $files; do cp "$saved/$f" "$f"; done
exit "$status"
