#!/bin/sh
# The 2048 render of run-script.sh, with the tools in Debug so a crash prints its stack trace.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
bin=/work/out/tensorfold-cuda-$(git -c safe.directory='*' rev-parse HEAD)
out=/work/out/h3-image
H3_BUILD=sm121 H3_OPT=Debug sh docs/recipes/minimax-h3-cuda/cuda-build.sh
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$out"
case=$out/2048
log=$case.prompt.log
"$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$case.safetensors" 2048x2048 42 2>"$log" || { cat "$log" >&2; exit 1; }
"$bin/tf-h3-image-cuda" "$model" "$case.safetensors" "$case"
