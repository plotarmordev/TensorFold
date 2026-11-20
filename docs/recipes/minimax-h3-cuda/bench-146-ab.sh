#!/bin/sh
# Card 146's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-146-base.patch reverse-applied — the card's kernel and family files back to
# main's — and bench-script.sh again: the baseline is the same job's main code with this commit's
# bench script, model paths and int8 default.
# Run 1 renders the pilot's fox prompt cold, the picture the episode-1 gates are read on (the Mac
# references are copied beside it for tools/zig/h3_image_compare.py on the fetched files); runs 2-4 are
# warm repeats on the three prompts the repo and earlier jobs never used (./prompts/tram|greenhouse|trench.txt).
# The TMA product claims bit-identical latents to the cuBLASLt + h3_dequantize path it replaces (the
# same products and roundings in registers), so every warm latent must be sha256-equal across phases.
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
# the episode-1 Mac picture and latent beside run 1 (the fox): the picture gates are
# tools/zig/h3_image_compare.py's run on the fetched /work/out, as card 122's
find /assets/drama3-pilot \( -name 2048-warm1.png -o -name t2i-2048.latent.f32 \) >/work/out/mac-refs.log
for want in 2048-warm1.png t2i-2048.latent.f32; do
    hit=$(grep "/$want\$" /work/out/mac-refs.log | head -1)
    if [ -n "$hit" ]; then cp "$hit" "/work/out/h3-image-bench/mac-$want"; else echo "no $want in the pilot folder"; fi
done
echo "=== before: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_PROFILE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
report /work/out/h3-image-bench-base
git -c safe.directory='*' apply "$patch_file"
echo "=== warm latents and PNGs, after then before ==="
for f in 2048-2 2048-3 2048-4; do
    for ext in latent.f32 png; do
        a=/work/out/h3-image-bench/$f.$ext
        b=/work/out/h3-image-bench-base/$f.$ext
        [ -f "$a" ] && [ -f "$b" ] || continue
        if cmp -s "$a" "$b"; then echo "$f.$ext identical"; else echo "$f.$ext DIFFERENT"; fi
    done
done
exit "$status"
