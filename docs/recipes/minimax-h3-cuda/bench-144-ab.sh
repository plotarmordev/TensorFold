#!/bin/sh
# Card 144's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-144-base.patch reverse-applied — the card's decoder files back to main's — and
# bench-script.sh again. The decoder doesn't depend on the prompt, so both phases render the canonical
# (seen) pilot prompt: run 1 cold, runs 2-4 warm kept-states hits, H3_PROFILE on.
# This commit's phase also runs bench-script.sh's same-latent `reference` compare (the decoder's
# full-chain bound); every run's latent must be byte-identical between the phases (the transformer is
# untouched). TOP has no PIL: the episode-1 gates run on the fetched pictures with tools/zig/h3_image_compare.py.
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-144-ab.sh
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-144-base.patch
status=0
stages() { # every run's H3_PROFILE decoder stage lines
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run decoder stages"
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.decoder.log" || true
    done
}
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
H3_PROFILE=1 sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
stages /work/out/h3-image-bench
grep -h "relative\|PSNR" /work/out/h3-image-bench/2048-reference*.log || true
echo "=== before: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_PROFILE=1 H3_BENCH_SKIP_COMPARE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
stages /work/out/h3-image-bench-base
echo "=== latents, this commit against main ==="
for run in 1 2 3 4; do
    if cmp -s /work/out/h3-image-bench/2048-$run.latent.f32 /work/out/h3-image-bench-base/2048-$run.latent.f32; then
        echo "2048-$run.latent.f32 identical"
    else
        echo "2048-$run.latent.f32 DIFFERENT"
        status=1
    fi
done
exit "$status"
