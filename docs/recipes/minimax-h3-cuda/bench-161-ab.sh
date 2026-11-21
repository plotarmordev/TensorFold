#!/bin/sh
# Card 161's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-161-base.patch reverse-applied (the products' algorithm picks back to main's) and the
# same runs again, then both again in fast mode (--draft-steps 3). Each phase renders the pilot prompt cold
# (episode 1's gate prompt: its cold start times every candidate, and the server log lists heuristic #1
# against the timed pick in TOPS) and three unseen prompts warm, H3_PROFILE on for the stage times.
# TOP has no PIL: the episode-1 gates run on the fetched 2048-1 picture and latent with tools/zig/h3_image_compare.py.
# Results: /work/out/h3-161-{card,main}{,-fast} and SUMMARY lines on stdout.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-161-ab.sh
# Run at 34cb7c3 (job top-9d39affb92e3): the timed picks lost and are reverted; the patch is the receipt of what ran.
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot/h3-image-pilot
p=/work/h3-prompts-161
mkdir -p $p
cat >$p/harbour.txt <<'EOF'
A small fishing harbour at first light, wooden boats with peeling blue and red paint tied to a stone
quay, coils of rope and stacked lobster pots, gulls on the bollards, mist lifting off flat water,
soft pastel sky, photorealistic, 50mm.
EOF
cat >$p/library.txt <<'EOF'
The reading room of an old university library at night: green-shaded brass lamps on long oak tables,
open books and a fountain pen, tall shelves climbing to a coffered ceiling, a rolling ladder, rain on
the arched windows, warm light and deep shadows, shot on medium format film.
EOF
cat >$p/market.txt <<'EOF'
A crowded spice market in a covered stone arcade at midday, pyramids of turmeric, paprika and cumin in
woven baskets, hanging brass lanterns, shafts of sunlight through gaps in the canvas roof, a vendor
weighing saffron on a hand scale, rich colour, natural light, high detail.
EOF
prompts="$pilot/prompt.txt $p/harbour.txt $p/library.txt $p/market.txt"
status=0
stages() { # every run's wall split and the transformer's H3_PROFILE block stages
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run"
        cat "$1/2048-$run.time" | tr '\n' ' ' | awk '{printf "wall %.2f s\n", $2 - $1}'
        grep -hE "^prompt: |^transformer: |s in all;" "$1/2048-$run.prompt.log" "$1/2048-$run.image.log" "$1/2048-$run.decoder.log" 2>/dev/null || true
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.image.log" || true
    done
}
bench() { # bench OUT [DRAFT_STEPS [SKIP_COMPARE]]: the four runs into /work/out/OUT
    H3_BENCH_OUT=/work/out/$1 H3_PROFILE=1 H3_DRAFT_STEPS=${2:-} H3_BENCH_SKIP_COMPARE=${3:-} H3_BENCH_PROMPTS="$prompts" \
        sh $here/bench-script.sh || status=$?
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
bench h3-161-card
bench h3-161-card-fast 3 skip
grep -h "product (" /work/out/h3-161-card/2048.server.log || true
echo "=== main: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R $here/bench-161-base.patch
bench h3-161-main "" skip
bench h3-161-main-fast 3 skip
git -c safe.directory='*' apply $here/bench-161-base.patch
for side in card main card-fast main-fast; do
    echo "=== $side ==="
    stages /work/out/h3-161-$side
done
grep -h "relative\|PSNR" /work/out/h3-161-card/2048-reference*.log || true
echo "=== latents, card against main (int8 sums are exact; a bf16 pick may round differently) ==="
for run in 1 2 3 4; do
    cmp -s /work/out/h3-161-card/2048-$run.latent.f32 /work/out/h3-161-main/2048-$run.latent.f32 \
        && echo "2048-$run identical" || echo "2048-$run differs"
done
exit "$status"
