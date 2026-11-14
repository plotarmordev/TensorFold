#!/bin/sh
# MiniMax-H3-Image's CUDA pictures on a Spark, through the Desk's GPU door inside a DeepSeek pause window:
#   ssh top-menu "job start gpu bottom plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/run-script.sh"
# /assets/drama3-pilot (read-only) holds models/MiniMax-H3-Image, its text encoder in text_encoder/, and
# h3-image-pilot/: prompt.txt and bin-SHA/, this commit's three tools, built beforehand where nvcc and g++ exist:
#   zig build tf-h3-image-prompt-host tf-h3-image-cuda tf-h3-image-decoder-host -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast
# Seed 42 at 2048x2048 and 4096x4096, the Mac's settings; each stage's log, with its time, lands in /work/out/h3-image
# beside the case, the latent and the PNG. The PSNR against the Mac's pictures is taken after `job fetch`.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
bin=$pilot/bin-$(git -c safe.directory='*' rev-parse HEAD)
out=/work/out/h3-image
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
for path in "$model/model_index.json" "$model/text_encoder/config.json" "$pilot/prompt.txt" "$bin/tf-h3-image-cuda"; do
    [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
mkdir -p "$out"
stage() { # stage NAME TOOL ARGS...: the tool's log, with its times, to $case.NAME.log; shown if it fails
    log=$case.$1.log
    shift
    "$@" 2>"$log" || { cat "$log" >&2; exit 1; }
}
for side in 2048 4096; do
    case=$out/$side
    stage prompt "$bin/tf-h3-image-prompt-host" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$case.safetensors" "${side}x$side" 42
    stage image "$bin/tf-h3-image-cuda" "$model" "$case.safetensors" "$case"
    stage decoder "$bin/tf-h3-image-decoder-host" "$model" "$case.latent.f32" $((side / 16)) $((side / 16)) "$case.png" "${side}x$side"
    echo "$side done"
done
