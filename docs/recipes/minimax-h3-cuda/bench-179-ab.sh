#!/bin/sh
# Card 179's same-job A/B: this commit's dense int8 attention, then bench-179-base.patch
# reverse-applied for main's bf16 attention. Each phase draws the pilot fox cold and three unseen
# prompts warm with H3_PROFILE stage times. Every card picture must pass gate.py against main's same-prompt
# reference; the card also runs the host-reference decode. Results: /work/out/h3-179-{card,main}.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-179-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-179-base.patch
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt $here/prompts/meadow.txt $here/prompts/steam.txt"
status=0
phase() { # phase NAME SKIP_COMPARE
    H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-179-$1 H3_BENCH_SKIP_COMPARE=$2 H3_BENCH_PROMPTS="$prompts" \
        sh "$here/bench-script.sh" || status=$?
}
echo "=== card: dense int8 QK and PV at $(git -c safe.directory='*' rev-parse --short HEAD) ==="
phase card ""
echo "=== main: bf16 attention (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
tree=$(git -c safe.directory='*' add -A zig && git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = 1fe941dfcf0ca9991e18a31f613eb20d12af1d50 ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
phase main 1
git -c safe.directory='*' apply "$patch_file"
for run in 1 2 3 4; do
    result=$(python3 "$here/gate.py" /work/out/h3-179-main/2048-$run /work/out/h3-179-card/2048-$run)
    echo "$result"
    case $result in *": PASS") ;; *) status=1 ;; esac
done
for side in card main; do
    echo "=== $side selftests ==="
    cat /work/out/h3-179-$side/selftests || :
    echo "=== $side stage times ==="
    for run in 1 2 3 4; do
        echo "--- 2048-$run"
        grep -hE '^  [a-z ]+: [0-9.]+ s$' /work/out/h3-179-$side/2048-$run.image.log || :
    done
done
grep -h "picture:" /work/out/h3-179-card/2048-reference.log || :
exit "$status"
