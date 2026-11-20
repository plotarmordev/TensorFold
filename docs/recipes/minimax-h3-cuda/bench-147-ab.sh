#!/bin/sh
# Card 147's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit with H3_LINEAR=fp4 — every transformer projection block-scaled NVFP4 on the FP4 tensor cores —
# then bench-147-base.patch reverse-applied — the card's files back to main's — and bench-script.sh
# again: the baseline is the same job's main code with this commit's bench script, model paths and
# tensor-core decoder default. Both phases run the pilot fox cold (a kept-states hit, the labelled
# cold run), then the three unseen prompts warm, H3_PROFILE stage times on every run; only the first
# phase decodes and gates the reference picture.
# Results: /work/out/h3-147-fp4 (this commit, --fp4) and /work/out/h3-147-base (the card's files at main).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-147-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-147-base.patch
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt $here/prompts/meadow.txt $here/prompts/steam.txt"
status=0
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD), H3_LINEAR=fp4 ==="
H3_LINEAR=fp4 H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-147-fp4 H3_BENCH_PROMPTS="$prompts" \
    sh "$here/bench-script.sh" || status=$?
echo "=== before: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-147-base H3_BENCH_SKIP_COMPARE=1 H3_PROFILE=1 H3_BENCH_PROMPTS="$prompts" \
    sh "$here/bench-script.sh" || status=$?
git -c safe.directory='*' apply "$patch_file"
exit "$status"
