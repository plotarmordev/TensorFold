#!/bin/sh
# MiniMax-H3-Image's first CUDA picture on a Spark, through the Desk's GPU door inside a DeepSeek pause window:
#   ssh top-menu "job start gpu bottom plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/run-script.sh"
# The pilot folder at /assets/h3-image-pilot (read-only) holds the MiniMax-H3-Image folder (model/), Qwen3-VL's text
# encoder (text-encoder/), the Mac's picture of the same prompt and seed (mac.png, from prompt.txt, 2048x2048, seed 42)
# and bin-SHA/: this commit's three tools, built beforehand where nvcc and g++ exist:
#   zig build tf-h3-image-prompt-host tf-h3-image-cuda tf-h3-image-decoder-host -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast
# Everything it writes lands in /work/out/h3-image: the case, the latent, the PNG, each stage's time and the PSNR.
set -eu
pilot=/assets/h3-image-pilot
bin=$pilot/bin-$(git -c safe.directory='*' rev-parse HEAD)
out=/work/out/h3-image
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
for path in "$pilot/model/model_index.json" "$pilot/text-encoder/config.json" "$pilot/prompt.txt" "$pilot/mac.png" "$bin/tf-h3-image-cuda"; do
    [ -e "$path" ] || { echo "missing $path; /assets holds:" >&2; find /assets -maxdepth 3 >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
mkdir -p "$out"
"$bin/tf-h3-image-prompt-host" "$pilot/text-encoder" "$pilot/model" "$pilot/prompt.txt" "$out/case.safetensors" 2048x2048 42
"$bin/tf-h3-image-cuda" "$pilot/model" "$out/case.safetensors" "$out/case"
"$bin/tf-h3-image-decoder-host" "$pilot/model" "$out/case.latent.f32" 128 128 "$out/case.png" 2048x2048
python3 tools/zig/moss_image_case.py compare "$out/case.png" "$pilot/mac.png" "$out/side.png" | tee "$out/psnr.json"
