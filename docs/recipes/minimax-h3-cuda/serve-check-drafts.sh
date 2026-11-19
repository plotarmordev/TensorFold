#!/bin/sh
# Bisect for the warm-serve crash: two plain serves with phase markers in the log.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/serve-check-drafts.sh"
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=/work/out/h3-serve-check-drafts
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
rm -rf "$out"
mkdir -p "$out"
"$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$out/c.safetensors" 2048x2048 42 262144
printf '%s\n' "$out/c.safetensors $out/p1" "$out/c.safetensors $out/p2" |
    "$bin/tf-h3-image-cuda" "$model" --serve >"$out/replies-p" 2>"$out/serve-p.log" || true
cat "$out/serve-p.log"
exit 0
