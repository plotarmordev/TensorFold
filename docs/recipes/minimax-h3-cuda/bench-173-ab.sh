#!/bin/sh
# Card 173's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit — the prompt encoder's products int8 at a scale per 128 of K on the TMA product — then
# bench-173-base.patch reverse-applied (zig/ back to main's bf16 encoder) and bench-script.sh again. Each
# phase renders the pilot fox cold, then three unseen prompts warm, H3_PROFILE stage times on every run.
# Then main's prompt tool and this commit's with --bf16 write each prompt's case: they must be the same
# bytes (--bf16 keeps main's encoder). No phase runs the host-reference decode: the picture gates are
# h3_image_compare.py's run on the fetched files.
# Results: /work/out/h3-173-{card,main,bf16}.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-173-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-173-base.patch
model=/assets/drama3-pilot/models/MiniMax-H3-Image
bin=/work/out/tensorfold-cuda-$(git -c safe.directory='*' rev-parse HEAD) # each phase's build overwrites it
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/greenhouse.txt $here/prompts/tram.txt $here/prompts/trench.txt"
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} # --bf16's cuBLASLt
status=0
cases() { # cases DIR [--bf16]: the phase's freshly built prompt tool writes each prompt's case, one process each
    mkdir -p "$1"
    run=0
    for prompt in $prompts; do
        run=$((run + 1))
        "$bin/tf-h3-image-prompt-cuda" ${2:-} "$model/text_encoder" "$model" "$prompt" "$1/2048-$run.safetensors" 2048x2048 42 \
            >"$1/2048-$run.log" 2>&1 || { status=1; tail -n 5 "$1/2048-$run.log"; }
    done
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD), int8 prompt encoder ==="
H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-173-card H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" sh "$here/bench-script.sh" || status=$?
cases /work/out/h3-173-bf16 --bf16
echo "=== main: the card's files at main (patch -R), bf16 prompt encoder ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_PROFILE=1 H3_BENCH_OUT=/work/out/h3-173-main H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" sh "$here/bench-script.sh" || status=$?
cases /work/out/h3-173-main-cases
git -c safe.directory='*' apply "$patch_file"
echo "=== --bf16 against main's prompt cases, byte for byte ==="
for run in 1 2 3 4; do
    cmp "/work/out/h3-173-bf16/2048-$run.safetensors" "/work/out/h3-173-main-cases/2048-$run.safetensors" \
        && echo "2048-$run case identical" || status=1
done
grep -h "text encoder\|tokens;" /work/out/h3-173-card/*.prompt.log /work/out/h3-173-main/*.prompt.log || :
exit "$status"
