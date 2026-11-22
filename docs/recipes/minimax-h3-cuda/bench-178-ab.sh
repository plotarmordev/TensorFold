#!/bin/sh
# Card 178's same-job int8 A/B: this commit, then bench-178-base.patch reverse-applied so zig/ is
# exactly the branch's merge parent (main at a03e5c459). Each phase runs the pilot fox cold, then
# fox, harbour, meadow and steam three times each warm; H3_PROFILE is on throughout. The fused
# epilogue must reproduce main's latents and PNGs byte for byte.
# Results: /work/out/h3-178-{heads,base}.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-178-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-178-base.patch
fox=/assets/drama3-pilot/h3-image-pilot/prompt.txt
harbour=$here/prompts/harbour.txt
meadow=$here/prompts/meadow.txt
steam=$here/prompts/steam.txt
prompts="$fox $fox $fox $fox $harbour $harbour $harbour $meadow $meadow $meadow $steam $steam $steam"
status=0
phase() {
    H3_LINEAR=int8 H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-178-$1 H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" \
        sh "$here/bench-script.sh" || status=$?
}
stages() { # every run's wall split and H3_PROFILE block stages
    for run in $(seq 1 13); do
        echo "--- $1 2048-$run"
        cat "$1/2048-$run.time" | tr '\n' ' ' | awk '{printf "wall %.2f s\n", $2 - $1}'
        grep -hE "^prompt: |^transformer: |s in all;" "$1/2048-$run.prompt.log" "$1/2048-$run.image.log" "$1/2048-$run.decoder.log" 2>/dev/null || true
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.image.log" || true
    done
}
sh "$here/check-178.sh"
echo "=== heads: this commit $(git -c safe.directory='*' rev-parse --short HEAD), int8 qkv + fused heads ==="
phase heads
stages /work/out/h3-178-heads
echo "=== base: main at a03e5c459, int8 qkv + h3_heads ==="
git -c safe.directory='*' apply -R "$patch_file"
tree=$(git -c safe.directory='*' add -A zig && git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = 9fa1f41ca6c1029251ad09b47f36424f3cc5875d ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
phase base
stages /work/out/h3-178-base
git -c safe.directory='*' apply "$patch_file"
echo "=== fused int8 against main, byte for byte ==="
for run in 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    for f in latent.f32 png; do
        cmp "/work/out/h3-178-heads/2048-$run.$f" "/work/out/h3-178-base/2048-$run.$f" && echo "2048-$run.$f identical" || status=1
    done
done
exit "$status"
