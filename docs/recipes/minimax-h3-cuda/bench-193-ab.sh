#!/bin/sh
# Card 193's same-job A/B: bench-script.sh on this commit (two banded passes in flight, and every decoder product on
# a cuBLASLt algorithm that fuses the bias epilogue, so no h3d_finish pass), then bench-193-base.patch reverse-applied
# (main's decoder) and the same runs again: the fox cold, then three unseen prompts warm, 2048², H3_PROFILE off. Per
# run: wall and the decoder's passes, GPU and in-all seconds. Every latent must be byte-identical between the sides;
# the PNGs differ by the products' accumulation order, gated by h3_image_compare.py on the fetched files (TOP has no
# PIL). The card's side also runs the same-latent `reference` compare; the job ends with the h3d_flash contract gates.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-193-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot/h3-image-pilot
p=/work/h3-prompts-193
mkdir -p $p
cat >$p/lighthouse.txt <<'EOF'
A lighthouse keeper's kitchen at dawn, a kettle steaming on a cast-iron stove, oilskins on a peg, a brass
barometer, salt-streaked windows over a grey sea, soft window light, photorealistic.
EOF
cat >$p/market.txt <<'EOF'
A floating market at sunrise, wooden boats heaped with mangoes and lotus flowers, women in straw hats,
mist over the canal, wooden stilt houses, warm golden light, 50mm documentary photograph.
EOF
cat >$p/observatory.txt <<'EOF'
An old observatory dome opened to a starry sky, a brass refracting telescope on its iron pier, star
charts pinned to the curved wall, a red lamp on the desk, long exposure, quiet and detailed.
EOF
prompts="$pilot/prompt.txt $p/lighthouse.txt $p/market.txt $p/observatory.txt"
status=0
runs() { # every run's wall and the decoder's passes, GPU and in-all seconds
    for run in 1 2 3 4; do
        wall=$(tr '\n' ' ' <"$1/2048-$run.time" | awk '{printf "%.2f", $2 - $1}')
        grep -h "s in all;" "$1/2048-$run.decoder.log" | sed 's/[(),]/ /g' |
            awk -v r=$run -v w="$wall" '{printf "2048-%d wall %s s; decoder passes %.2f s, GPU %.2f s, in all %.2f s\n", r, w, $7, $9, $14}'
    done
}
bench() { # bench OUT [SKIP_COMPARE]: the four runs into /work/out/OUT
    H3_BENCH_OUT=/work/out/$1 H3_PROFILE= H3_BENCH_SKIP_COMPARE=${2:-} H3_BENCH_PROMPTS="$prompts" \
        sh $here/bench-script.sh || status=$?
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
bench h3-193-card
echo "=== main: the decoder driver at main (patch -R) ==="
git -c safe.directory='*' apply -R $here/bench-193-base.patch
bench h3-193-main skip
git -c safe.directory='*' apply $here/bench-193-base.patch
for side in card main; do
    echo "=== $side ==="
    runs /work/out/h3-193-$side
done
echo "=== same-latent reference (card) ==="
tail -n 8 /work/out/h3-193-card/2048-reference.compare.log || true
echo "=== latents and pictures, card against main ==="
for run in 1 2 3 4; do
    cmp -s /work/out/h3-193-card/2048-$run.latent.f32 /work/out/h3-193-main/2048-$run.latent.f32 && l=same || { l=DIFFERS; status=1; }
    cmp -s /work/out/h3-193-card/2048-$run.png /work/out/h3-193-main/2048-$run.png && p=identical || p=differs
    echo "2048-$run latent $l, png $p"
done
echo "=== tf-cuda-test kernel-contract h3d_flash ==="
H3_BUILD=contract H3_CONTRACT=h3d_flash sh $here/cuda-build.sh >/work/out/contract-193.log 2>&1 || status=$?
grep -E "gate|PASS|FAIL|exit" /work/out/contract-193.log | grep -v warning | tail -n 20
exit "$status"
