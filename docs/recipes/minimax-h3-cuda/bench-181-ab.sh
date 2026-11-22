#!/bin/sh
# Card 181's same-job A/B (job clones carry no history to check main out from): the h3 kernels' contract gates
# on this commit (h3_norm and h3_gate_add read the bf16 tables), then bench-script.sh on this commit — the
# AdaLN tables stored bf16 and the fused norms' weight and table reads 8 bytes a lane — then
# bench-181-base.patch reverse-applied (the job stops unless the result is main's zig/ tree) and the same runs
# again. Each phase renders the pilot fox cold, then three prompts neither phase's server has seen warm,
# H3_PROFILE on for the stage times. The card's latents and PNGs must be main's bytes: the job fails when any
# differs, so main's episode-1 gates (h3_image_compare.py on the fetched fox) carry over unchanged.
# Results: /work/out/h3-181-{card,main} and the stage lines on stdout.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-181-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-181-base.patch
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/greenhouse.txt $here/prompts/tram.txt $here/prompts/trench.txt"
status=0
echo "=== contract: h3 kernels at this commit ==="
H3_BUILD=contract H3_CONTRACT=h3 sh "$here/cuda-build.sh" || status=$?
phase() { # phase NAME: bench-script.sh into /work/out/h3-181-NAME
    H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-181-$1 H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" \
        sh "$here/bench-script.sh" || status=$?
}
stages() { # every run's wall, its prompt/transformer/decoder lines and the block stages
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run"
        tr '\n' ' ' <"$1/2048-$run.time" | awk '{printf "wall %.2f s\n", $2 - $1}'
        grep -hE "^prompt: |^transformer: |s in all;" "$1/2048-$run.prompt.log" "$1/2048-$run.image.log" "$1/2048-$run.decoder.log" 2>/dev/null || true
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.image.log" || true
    done
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
phase card
echo "=== main: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
tree=$(git -c safe.directory='*' add -A zig && git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = e98644533d4fdd7e6aea4db44bd2143af2d9b77c ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
phase main
git -c safe.directory='*' apply "$patch_file"
for side in card main; do
    echo "=== $side ==="
    cat "/work/out/h3-181-$side/selftests"
    stages "/work/out/h3-181-$side"
done
echo "=== card against main, byte for byte ==="
for run in 1 2 3 4; do
    for f in latent.f32 png; do
        cmp "/work/out/h3-181-card/2048-$run.$f" "/work/out/h3-181-main/2048-$run.$f" && echo "2048-$run.$f identical" || status=1
    done
done
exit "$status"
