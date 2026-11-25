#!/bin/sh
# Card 178's same-job int8 A/B, one prompt a job so each stays inside the GPU slot's 600 s. TOP gives each job a
# fresh /work except a zig-cache that survives while the slot stays with this card, so the baseline builds first
# at full cost and the heads phase re-applies the card's patch and rebuilds into its own bin dir — zig reuses
# every object the patch did not touch, and each side's tools land in a phase dir so the commit-keyed default
# could never serve one side's binaries to the other. bench-178-base.patch's reverse applies to zig/ to
# reconstruct main at 6b6822bbd; its zig tree is checked against 8a06c4029fff70b09d7e9286813975262ed0bd89. Each
# side renders the prompt cold, then three times warm with H3_PROFILE on; the fused epilogue must reproduce
# main's latents and PNGs byte for byte. check-178.sh's own job holds the contract gates. Run through
# bench-178-{fox,harbour,meadow,steam}.sh:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-178-fox.sh
set -eu
[ -n "${H3_178_TIMED:-}" ] || H3_178_TIMED=1 exec timeout 600 sh "$0"
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-178-base.patch
case $H3_178_PROMPT in
fox) file=/assets/drama3-pilot/h3-image-pilot/prompt.txt ;;
*) file=$here/prompts/$H3_178_PROMPT.txt ;;
esac
status=0
phase() { # $1 heads|base; each side builds into its own dir or the same key would reuse the other side's tools
    case $1 in
    heads) bindir=/work/out/h3-178-bin-heads ;;
    base) bindir=/work/out/h3-178-bin-base ;;
    esac
    H3_BENCH_BIN=$bindir H3_LINEAR=int8 H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-178-$H3_178_PROMPT-$1 \
        H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$file $file $file $file" \
        sh "$here/bench-script.sh" || status=$?
}
stages() { # every run's wall split and H3_PROFILE block stages
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run"
        tr '\n' ' ' <"$1/2048-$run.time" | awk '{printf "wall %.2f s\n", $2 - $1}'
        grep -hE "^prompt: |^transformer: |s in all;" "$1/2048-$run.prompt.log" "$1/2048-$run.image.log" "$1/2048-$run.decoder.log" 2>/dev/null || true
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.image.log" || true
    done
}
# a prior job killed mid-A/B can leave zig/ on the baseline; start clean so 'apply -R' is the only patch
git -c safe.directory='*' checkout -- zig/ 2>/dev/null || :
echo "=== base ($H3_178_PROMPT): main at 6b6822bbd, int8 qkv + h3_heads ==="
git -c safe.directory='*' apply -R "$patch_file"
tmp_index=$(mktemp)
trap 'git -c safe.directory="*" apply "$patch_file"; rm -f "$tmp_index"' EXIT
GIT_INDEX_FILE=$tmp_index git -c safe.directory='*' read-tree HEAD
tree=$(GIT_INDEX_FILE=$tmp_index git -c safe.directory='*' add -A zig && GIT_INDEX_FILE=$tmp_index git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = 8a06c4029fff70b09d7e9286813975262ed0bd89 ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
git -c safe.directory='*' diff --cached --quiet || { echo "real index touched by staged baseline" >&2; exit 1; }
phase base
git -c safe.directory='*' apply "$patch_file"
trap 'rm -f "$tmp_index"' EXIT
echo "=== heads ($H3_178_PROMPT): this commit $(git -c safe.directory='*' rev-parse --short HEAD), int8 qkv + fused heads ==="
phase heads
for side in heads base; do stages /work/out/h3-178-$H3_178_PROMPT-$side; done
echo "=== fused int8 against main, byte for byte ==="
for run in 1 2 3 4; do
    for f in latent.f32 png; do
        cmp "/work/out/h3-178-$H3_178_PROMPT-heads/2048-$run.$f" "/work/out/h3-178-$H3_178_PROMPT-base/2048-$run.$f" &&
            echo "2048-$run.$f identical" || status=1
    done
done
exit "$status"
