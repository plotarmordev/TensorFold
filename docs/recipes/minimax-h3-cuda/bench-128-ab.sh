#!/bin/sh
# Card 128's same-job A/B at head: bench-script.sh on this commit, then the card's six files checked out from
# main and bench-script.sh again — the baseline is the same job's main code with this commit's bench script,
# model paths and --tc decoder default. `cuda_fused_test.zig` is the card's own file: it goes away for the
# baseline phase and back for the exit.
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (the card's files at main).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-128-ab.sh"
# main in a job clone: no local branch, no origin/main, often a shallow detached HEAD with no merge
# commits. Fetching main is the reliable path; the shallowest fallbacks are a merge's second parent
# (when the clone kept the merge) and refs/heads/main.
main=$(git -c safe.directory='*' merge-base HEAD origin/main 2>/dev/null || true)
if [ -z "$main" ]; then
    git -c safe.directory='*' fetch origin main 2>/dev/null || true
    main=$(git -c safe.directory='*' rev-parse FETCH_HEAD 2>/dev/null || true)
fi
if [ -z "$main" ]; then
    m=$(git -c safe.directory='*' rev-list --merges -1 HEAD 2>/dev/null || true)
    [ -n "$m" ] && main=$(git -c safe.directory='*' rev-parse "$m^2" 2>/dev/null || true)
fi
[ -n "$main" ] || main=refs/heads/main
status=0
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD), card files at base ${main} ==="
H3_PROFILE=1 sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
echo "=== before: the card's six files at main ==="
# main has no cuda_fused_test.zig: checkout the five shared files from main, remove the card's file
git -c safe.directory='*' checkout "$main" -- zig/kernels/cuda/h3.cu zig/src/core/contract/registry_media.zig zig/src/families/h3/cuda.zig zig/src/families/h3/cuda_int8.zig zig/tests/h3_image_cuda.zig || { echo "no main ref resolves; baseline skipped" >&2; exit 64; }
rm -f zig/src/families/h3/cuda_fused_test.zig
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_BENCH_SKIP_COMPARE=1 H3_PROFILE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' checkout HEAD -- zig/kernels/cuda/h3.cu zig/src/core/contract/registry_media.zig zig/src/families/h3/cuda.zig zig/src/families/h3/cuda_int8.zig zig/src/families/h3/cuda_fused_test.zig zig/tests/h3_image_cuda.zig
exit "$status"
