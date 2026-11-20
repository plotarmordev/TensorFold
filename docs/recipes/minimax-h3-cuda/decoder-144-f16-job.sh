#!/bin/sh
# Card 144's FP16 trial in one TOP window: the decoder kernels' contract gates, then bench-script.sh on this
# commit (bf16 products) and again with decoder-144-f16.patch applied (the --tc products on fp16 operands —
# weights and each product's input cast by h3d_half, counting values outside fp16's normals — fp32 sums into acc,
# then h3d_finish's roundings). Both phases render the canonical (seen) pilot prompt with H3_PROFILE on and run the
# decoder selftest and same-latent `reference`; every run's latent must be byte-identical between the phases.
# Results: /work/out/h3-image-bench (bf16) and /work/out/h3-image-bench-f16 (the trial).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/decoder-144-f16-job.sh
set -eu
patch_file=docs/recipes/minimax-h3-cuda/decoder-144-f16.patch
status=0
stages() { # every run's H3_PROFILE decoder stage lines and the trial's fp16 counts
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run decoder"
        grep -E '^  [a-z ]+: [0-9.]+ s$|FP16 trial' "$1/2048-$run.decoder.log" || true
    done
    grep -h "FP16 trial" "$1"/*decoder*server.log 2>/dev/null | sort | uniq -c || true
}
echo "=== contract: the h3 image decoder kernels ==="
sh docs/recipes/minimax-h3-cuda/contract-job.sh || status=$?
for side in bf16 f16; do
    out=/work/out/h3-image-bench
    if [ "$side" = f16 ]; then
        git -c safe.directory='*' apply "$patch_file"
        out=/work/out/h3-image-bench-f16
    fi
    echo "=== $side products ==="
    H3_BENCH_OUT=$out H3_PROFILE=1 sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
    stages "$out"
    cat "$out/selftests" || true
    grep -h "relative\|PSNR\|FAIL\|FP16 trial" "$out"/2048-reference*.log "$out"/selftest-decoder.log || true
done
git -c safe.directory='*' apply -R "$patch_file"
echo "=== latents, f16 against bf16 ==="
for run in 1 2 3 4; do
    cmp -s /work/out/h3-image-bench/2048-$run.latent.f32 /work/out/h3-image-bench-f16/2048-$run.latent.f32 &&
        echo "2048-$run.latent.f32 identical" || { echo "2048-$run.latent.f32 DIFFERENT"; status=1; }
done
exit "$status"
