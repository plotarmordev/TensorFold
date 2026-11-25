#!/bin/sh
# MiniMax-H3-Image's CUDA picture on TOP's Spark, through the Desk's GPU door:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/run-script.sh"
# /assets/drama3-pilot (read-only) holds models/MiniMax-H3-Image, its text encoder in text_encoder/, and
# h3-image-pilot/prompt.txt. cuda-build.sh unpacks the staged toolchain and builds this commit's tools.
# Seed 42 at 2048x2048, the Mac's settings; each stage's log, with its time, lands in /work/out/h3-image
# beside the case, the latent and the PNG. The PSNR against the Mac's picture is taken after `job fetch`.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
bin=/work/out/tensorfold-cuda-$(git -c safe.directory='*' rev-parse HEAD)
out=/work/out/h3-image
H3_BUILD=sm121 H3_OPT=${H3_OPT:-ReleaseFast} sh docs/recipes/minimax-h3-cuda/cuda-build.sh
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
for path in "$model/model_index.json" "$model/text_encoder/config.json" "$pilot/prompt.txt" "$bin/tf-h3-image-cuda" "$bin/tf-h3-image-prompt-cuda" "$bin/tf-h3-image-decoder-cuda"; do
    [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
mkdir -p "$out"
failed=0
"$bin/tf-h3-image-cuda" --self-test || failed=1
"$bin/tf-h3-image-decoder-cuda" selftest || failed=1
"$bin/tf-h3-image-prompt-cuda" selftest /work/out/h3-prompt-selftest || failed=1
[ "$failed" -eq 0 ] || exit 1
stage() { # stage NAME TOOL ARGS...: the tool's log, with its times, to $case.NAME.log; shown if it fails
    log=$case.$1.log
    shift
    "$@" 2>"$log" || { cat "$log" >&2; exit 1; }
}
for side in 2048; do
    case=$out/$side
    stage prompt "$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$case.safetensors" "${side}x$side" 42
    stage image "$bin/tf-h3-image-cuda" "$model" "$case.safetensors" "$case"
    stage decoder "$bin/tf-h3-image-decoder-cuda" "$model" "$case.latent.f32" $((side / 16)) $((side / 16)) "$case.png" "${side}x$side"
    echo "$side done"
done
