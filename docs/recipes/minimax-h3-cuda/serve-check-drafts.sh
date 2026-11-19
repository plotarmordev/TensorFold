#!/bin/sh
# Bisect for the warm-serve crash: two plain serves under gdb for a backtrace. ~4 min after the
# build.
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
printf '%s\n' "$out/c.safetensors $out/p1" "$out/c.safetensors $out/p2" >"$out/requests"
if command -v gdb >/dev/null 2>&1; then
    gdb -batch -ex "handle SIGPIPE nostop noprint" -ex run -ex "bt 30" -ex "info args" -ex quit \
        --args "$bin/tf-h3-image-cuda" "$model" --serve <"$out/requests" >"$out/gdb.log" 2>&1 || true
    tail -80 "$out/gdb.log"
else
    echo "no gdb"; "$bin/tf-h3-image-cuda" "$model" --serve <"$out/requests"
fi
exit 0
