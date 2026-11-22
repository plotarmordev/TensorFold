#!/bin/sh
# Card 175's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-175-base.patch reverse-applied (card 144's wmma h3d_flash) and the same runs again.
# Run 1 renders the pilot's fox cold; runs 2-4 are warm on three prompts no earlier job used
# (./prompts/pier|kiln|dunes.txt), H3_PROFILE on for the decoder's stage seconds. The transformer is
# untouched, so every latent must be byte-equal across the phases; the same-latent `reference` runs on
# this commit's phase.
# Results: /work/out/h3-175-{card,main}.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-175-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
export H3_BENCH_PROMPTS="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/pier.txt $here/prompts/kiln.txt $here/prompts/dunes.txt"
status=0
# report DIR: the selftests (a FAIL fails the job), then each run's wall seconds and its prompt, image and decoder logs
report() {
    cat "$1/selftests"
    ! grep -q FAIL "$1/selftests" || status=1
    for run in 1 2 3 4; do
        case=$1/2048-$run
        echo "--- run $run: $(awk 'NR == 1 { s = $1 } END { printf "%.2f s wall", $1 - s }' "$case.time")"
        cat "$case.prompt.log" "$case.image.log" "$case.decoder.log"
    done
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
H3_BENCH_OUT=/work/out/h3-175-card H3_PROFILE=1 sh $here/bench-script.sh || status=$?
report /work/out/h3-175-card
grep -h "relative\|PSNR" /work/out/h3-175-card/2048-reference*.log || status=1
echo "=== main: the card's kernel at main (patch -R) ==="
git -c safe.directory='*' apply -R $here/bench-175-base.patch
H3_BENCH_OUT=/work/out/h3-175-main H3_PROFILE=1 H3_BENCH_SKIP_COMPARE=1 sh $here/bench-script.sh || status=$?
git -c safe.directory='*' apply $here/bench-175-base.patch
report /work/out/h3-175-main
echo "=== latents, card against main ==="
for run in 1 2 3 4; do
    if cmp -s /work/out/h3-175-card/2048-$run.latent.f32 /work/out/h3-175-main/2048-$run.latent.f32; then
        echo "2048-$run.latent.f32 identical"
    else
        echo "2048-$run.latent.f32 DIFFERENT"
        status=1
    fi
done
exit "$status"
