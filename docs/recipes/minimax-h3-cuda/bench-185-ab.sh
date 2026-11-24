#!/bin/sh
# Card 185's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this commit,
# then bench-185-base.patch reverse-applied (the decoder back to one whole-picture pass and png.encode) and the same
# runs again. Each phase renders the pilot prompt cold and three unseen prompts warm at 2048², H3_PROFILE off so no
# stage waits skew the decoder. Per run: wall, the decoder's pass wall, GPU and in-all seconds, and its host side
# (in all less the passes); then each picture's sha256 on both sides, which must match.
# Run at d3de0451 (job top-b20e9be30f8b): kept; the patch is the receipt of what ran.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-185-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot/h3-image-pilot
p=/work/h3-prompts-185
mkdir -p $p
cat >$p/glasshouse.txt <<'EOF'
A Victorian glasshouse in late afternoon, wrought-iron ribs and fogged panes, tree ferns and orchids
on tiered staging, a gardener's watering can on the tiled floor, dust in slanting sunbeams,
photorealistic, 35mm.
EOF
cat >$p/tram.txt <<'EOF'
An old yellow tram climbing a steep cobbled street on a rainy evening, reflections of shop signs in
the wet stones, people with umbrellas at the stop, overhead wires against a violet sky, cinematic.
EOF
cat >$p/workshop.txt <<'EOF'
A violin maker's workshop: unvarnished violin bodies hanging on hooks, curls of spruce shavings on a
worn bench, chisels and calipers, a window of north light, warm wood tones, macro detail.
EOF
prompts="$pilot/prompt.txt $p/glasshouse.txt $p/tram.txt $p/workshop.txt"
status=0
runs() { # every run's wall and the decoder's seconds, its host side the in-all less the passes' wall
    for run in 1 2 3 4; do
        wall=$(tr '\n' ' ' <"$1/2048-$run.time" | awk '{printf "%.2f", $2 - $1}')
        grep -h "s in all;" "$1/2048-$run.decoder.log" | sed 's/[(),]/ /g' |
            awk -v r=$run -v w="$wall" '{printf "2048-%d wall %s s; decoder passes %.2f s, GPU %.2f s, in all %.2f s, host side %.2f s\n", r, w, $7, $9, $14, $14 - $7}'
    done
}
bench() { # bench OUT [SKIP_COMPARE]: the four runs into /work/out/OUT
    H3_BENCH_OUT=/work/out/$1 H3_PROFILE= H3_BENCH_SKIP_COMPARE=${2:-} H3_BENCH_PROMPTS="$prompts" \
        sh $here/bench-script.sh || status=$?
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
bench h3-185-card skip
echo "=== main: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R $here/bench-185-base.patch
bench h3-185-main skip
git -c safe.directory='*' apply $here/bench-185-base.patch
for side in card main; do
    echo "=== $side ==="
    runs /work/out/h3-185-$side
done
echo "=== pictures, card against main (sha256) ==="
for run in 1 2 3 4; do
    c=$(sha256sum </work/out/h3-185-card/2048-$run.png | cut -d' ' -f1)
    m=$(sha256sum </work/out/h3-185-main/2048-$run.png | cut -d' ' -f1)
    cmp -s /work/out/h3-185-card/2048-$run.latent.f32 /work/out/h3-185-main/2048-$run.latent.f32 && l=same || l=differs
    [ "$c" = "$m" ] && echo "2048-$run identical $c (latent $l)" || { echo "2048-$run DIFFERS card $c main $m (latent $l)"; status=1; }
done
exit "$status"
