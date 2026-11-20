#!/bin/sh
# Card 144's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-144-base.patch reverse-applied — the card's decoder files back to main's — and
# bench-script.sh again: the baseline is the same job's main code with this commit's bench script and
# model paths.
# Each phase renders the pilot's prompt cold (the picture the episode-1 gates are read on) and three
# warm repeats on prompts the repo and earlier jobs never used, written to /work/h3-prompts here.
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main). The
# phases' warm latents and PNGs must be sha256-pairwise equal only where the code paths match — the
# transformer is untouched, so differing latents flag a bug, not noise.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-144-ab.sh"
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-144-base.patch
mkdir -p /work/h3-prompts
cat >/work/h3-prompts/unseen-1.txt <<'EOF'
A weathered lighthouse keeper's cottage on a black basalt shore at blue hour, warm lamplight in two
windows, sea stack silhouettes behind low fog, wet stone reflections, muted teal and amber palette,
photorealistic, 35mm.
EOF
cat >/work/h3-prompts/unseen-2.txt <<'EOF'
An overhead studio photograph of a watchmaker's bench: brass gears, a loupe, tweezers and an open
pocket watch on a green cutting mat, soft north light, shallow depth of field, fine dust visible.
EOF
cat >/work/h3-prompts/unseen-3.txt <<'EOF'
A night market alley in the rain: paper lanterns reflecting off wet pavement, a vendor closing a
noodle stall, steam rising, bicycles leaning on shutters, cinematic grain, deep shadows.
EOF
export H3_BENCH_PROMPTS="/work/h3-prompts/unseen-1.txt /work/h3-prompts/unseen-2.txt /work/h3-prompts/unseen-3.txt"
status=0
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
H3_PROFILE=1 sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
echo "=== before: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_PROFILE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
echo "=== latent/PNG digests, after then before ==="
for f in 2048-2 2048-3 2048-4; do
    for ext in latent.f32 png; do
        a=/work/out/h3-image-bench/$f.$ext
        b=/work/out/h3-image-bench-base/$f.$ext
        [ -f "$a" ] && [ -f "$b" ] || continue
        if cmp -s "$a" "$b"; then echo "$f.$ext identical"; else echo "$f.$ext DIFFERENT"; fi
    done
done
exit "$status"
