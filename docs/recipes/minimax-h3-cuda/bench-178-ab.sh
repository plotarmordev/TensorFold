#!/bin/sh
# Card 178's same-job int8 A/B: this commit, then bench-178-base.patch reverse-applied so zig/ is
# exactly the branch's merge parent (main at a03e5c459). Each phase runs the pilot fox cold, then the
# three unseen prompts warm; H3_PROFILE is on throughout. The fused epilogue must reproduce main's
# latents and PNGs byte for byte.
# Results: /work/out/h3-178-{heads,base}.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-178-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-178-base.patch
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt $here/prompts/meadow.txt $here/prompts/steam.txt"
status=0
phase() {
    H3_LINEAR=int8 H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-178-$1 H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" \
        sh "$here/bench-script.sh" || status=$?
}
echo "=== heads: this commit $(git -c safe.directory='*' rev-parse --short HEAD), int8 qkv + fused heads ==="
phase heads
echo "=== base: main at a03e5c459, int8 qkv + h3_heads ==="
git -c safe.directory='*' apply -R "$patch_file"
tree=$(git -c safe.directory='*' add -A zig && git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = 9fa1f41ca6c1029251ad09b47f36424f3cc5875d ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
phase base
git -c safe.directory='*' apply "$patch_file"
echo "=== fused int8 against main, byte for byte ==="
for run in 1 2 3 4; do
    for f in latent.f32 png; do
        cmp "/work/out/h3-178-heads/2048-$run.$f" "/work/out/h3-178-base/2048-$run.$f" && echo "2048-$run.$f identical" || status=1
    done
done
exit "$status"
