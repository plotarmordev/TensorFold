#!/bin/sh
# Card 178's same-job int8 A/B, one prompt a job so each stays inside the GPU slot's 600 s: this commit, then
# bench-178-base.patch reverse-applied so zig/ is exactly the branch's merge parent (main at 6b6822bbd). Each side
# renders the prompt cold, then three times warm; H3_PROFILE is on throughout. The fused epilogue must reproduce
# main's latents and PNGs byte for byte. The contract gates are check-178.sh's own job.
# Results: /work/out/h3-178-PROMPT-{heads,base}. Run through bench-178-{fox,harbour,meadow,steam}.sh:
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
phase() {
    H3_LINEAR=int8 H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-178-$H3_178_PROMPT-$1 H3_BENCH_SKIP_COMPARE=1 \
        H3_BENCH_PROMPTS="$file $file $file $file" sh "$here/bench-script.sh" || status=$?
}
stages() { # every run's wall split and H3_PROFILE block stages
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run"
        tr '\n' ' ' <"$1/2048-$run.time" | awk '{printf "wall %.2f s\n", $2 - $1}'
        grep -hE "^prompt: |^transformer: |s in all;" "$1/2048-$run.prompt.log" "$1/2048-$run.image.log" "$1/2048-$run.decoder.log" 2>/dev/null || true
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.image.log" || true
    done
}
echo "=== heads ($H3_178_PROMPT): this commit $(git -c safe.directory='*' rev-parse --short HEAD), int8 qkv + fused heads ==="
phase heads
echo "=== base ($H3_178_PROMPT): main at 6b6822bbd, int8 qkv + h3_heads ==="
git -c safe.directory='*' apply -R "$patch_file"
trap 'git -c safe.directory="*" apply "$patch_file"; rm -f "$tmp_index"' EXIT
tmp_index=$(mktemp)
GIT_INDEX_FILE=$tmp_index git -c safe.directory='*' read-tree HEAD
tree=$(GIT_INDEX_FILE=$tmp_index git -c safe.directory='*' add -A zig && GIT_INDEX_FILE=$tmp_index git -c safe.directory='*' write-tree --prefix=zig/)
[ "$tree" = 8a06c4029fff70b09d7e9286813975262ed0bd89 ] || { echo "patch -R rebuilt zig/ $tree, not main's" >&2; exit 1; }
git -c safe.directory='*' diff --cached --quiet || { echo "real index touched by staged baseline" >&2; exit 1; }
phase base
for side in heads base; do stages /work/out/h3-178-$H3_178_PROMPT-$side; done
echo "=== fused int8 against main, byte for byte ==="
for run in 1 2 3 4; do
    for f in latent.f32 png; do
        cmp "/work/out/h3-178-$H3_178_PROMPT-heads/2048-$run.$f" "/work/out/h3-178-$H3_178_PROMPT-base/2048-$run.$f" &&
            echo "2048-$run.$f identical" || status=1
    done
done
exit "$status"
