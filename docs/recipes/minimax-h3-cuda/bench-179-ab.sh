#!/bin/sh
# Card 179's same-job A/B: this commit's dense int8 attention, then bench-179-base.patch
# reverse-applied for main's bf16 attention. Each phase draws the pilot fox cold and three unseen
# prompts warm with H3_PROFILE stage times. Results: /work/out/h3-179-{card,main}.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-179-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-179-base.patch
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt $here/prompts/meadow.txt $here/prompts/steam.txt"
status=0
phase() {
    H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-179-$1 H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" \
        sh "$here/bench-script.sh" || status=$?
}
echo "=== card: dense int8 QK and PV at $(git -c safe.directory='*' rev-parse --short HEAD) ==="
phase card
echo "=== main: bf16 attention (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
tree=$(git -c safe.directory='*' add -A zig && git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = e98644533d4fdd7e6aea4db44bd2143af2d9b77c ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
phase main
git -c safe.directory='*' apply "$patch_file"
for side in card main; do
    echo "=== $side stage times ==="
    grep -h '^  attention ' /work/out/h3-179-$side/2048-*.image.log || :
done
exit "$status"
