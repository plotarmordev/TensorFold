#!/bin/sh
# Card 147's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit with H3_LINEAR=fp4 — every transformer projection block-scaled NVFP4 on the FP4 tensor cores —
# then this commit at its int8 default, then bench-147-base.patch reverse-applied — zig/ back to main's
# (b85186af; the job stops unless the result is main's zig/ tree) — and bench-script.sh again. Every phase
# runs the pilot fox cold (a kept-states hit, the labelled cold run), then the three unseen prompts warm,
# H3_PROFILE stage times on every run. The int8 phase's latents and PNGs must be main's bytes: the job
# fails when any differs. No phase runs the host-reference decode: the picture gates are
# h3_image_compare.py's run on the fetched files.
# Results: /work/out/h3-147-{fp4,int8,base}.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-147-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-147-base.patch
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt $here/prompts/meadow.txt $here/prompts/steam.txt"
status=0
phase() { # phase NAME LINEAR: bench-script.sh into /work/out/h3-147-NAME
    H3_LINEAR=$2 H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-147-$1 H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" \
        sh "$here/bench-script.sh" || status=$?
}
echo "=== fp4: this commit $(git -c safe.directory='*' rev-parse --short HEAD), H3_LINEAR=fp4 ==="
phase fp4 fp4
echo "=== int8: this commit, the default ==="
phase int8 int8
echo "=== base: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
tree=$(git -c safe.directory='*' add -A zig && git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = 511d808ba0837d38abd7042b89f3631103c2dff5 ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
phase base int8
git -c safe.directory='*' apply "$patch_file"
echo "=== the default path against main, byte for byte ==="
for run in 1 2 3 4; do
    for f in latent.f32 png; do
        cmp "/work/out/h3-147-int8/2048-$run.$f" "/work/out/h3-147-base/2048-$run.$f" && echo "2048-$run.$f identical" || status=1
    done
done
exit "$status"
