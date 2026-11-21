#!/bin/sh
# Card 146's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-146-base.patch reverse-applied — the card's kernel and family files back to
# main's — and bench-script.sh again: the baseline is the same job's main code with this commit's
# bench script, model paths and int8 default.
# Run 1 renders the pilot's fox prompt cold, the picture the episode-1 gates are read on; runs 2-4 are
# warm repeats on the three prompts the repo and earlier jobs never used (./prompts/tram|greenhouse|trench.txt).
# The TMA product claims bit-identical latents to the cuBLASLt + h3_dequantize path it replaces (the
# same products and roundings in registers), so every latent must be byte-equal across phases.
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-146-ab.sh"
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-146-base.patch
here=$(dirname "$0")
export H3_BENCH_PROMPTS="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/tram.txt $here/prompts/greenhouse.txt $here/prompts/trench.txt"
status=0
# report DIR: the selftests (a FAIL fails the job), then each run's wall seconds and H3_PROFILE stage lines
report() {
    cat "$1/selftests"
    ! grep -q FAIL "$1/selftests" || status=1
    for run in 1 2 3 4; do
        case=$1/2048-$run
        echo "--- run $run: $(awk 'NR == 1 { s = $1 } END { printf "%.2f s wall", $1 - s }' "$case.time")"
        cat "$case.prompt.log" "$case.image.log" "$case.decoder.log"
    done
}
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
H3_PROFILE=1 sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
report /work/out/h3-image-bench
echo "=== before: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_PROFILE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
report /work/out/h3-image-bench-base
git -c safe.directory='*' apply "$patch_file"
echo "=== latents and PNGs, after then before ==="
for f in 2048-1 2048-2 2048-3 2048-4; do
    for ext in latent.f32 png; do
        a=/work/out/h3-image-bench/$f.$ext
        b=/work/out/h3-image-bench-base/$f.$ext
        if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo "$f.$ext MISSING"; status=1
        elif cmp -s "$a" "$b"; then echo "$f.$ext identical"
        else echo "$f.$ext DIFFERENT"; status=1; fi
    done
done
exit "$status"
