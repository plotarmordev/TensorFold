#!/bin/sh
# mmh3 at 9c0652f on run-script.sh's prompt, seed, sizes and six distilled steps, for its stage times beside
# TensorFold's, through the Desk's GPU door inside a DeepSeek pause window:
#   ssh top-menu "job start gpu bottom plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/mmh3-job.sh"
# The job has no network: mmh3-stage.sh puts mmh3's source, its vendored crates and Rust in the pilot folder, and the
# job builds mmh3 from them. models/MiniMax-H3-Image-mmh3 holds the files of Comfy-Org/MiniMax-H3-Image (rev 0bb149f5)
# that mmh3 reads: diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors,
# text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors and vae/minimax_h3_image_vae_fp16.safetensors.
# Its CUDA linear layers are INT8, the one difference from TensorFold's BF16 run. mmh3 logs its prompt, load, step and
# decoder times; each picture's log, ending in its wall time, lands in /work/out/mmh3 beside the PNG.
set -eu
pilot=/assets/drama3-pilot
stage=$pilot/mmh3-9c0652f
models=$pilot/models/MiniMax-H3-Image-mmh3
prompt=$pilot/h3-image-pilot/prompt.txt
out=/work/out/mmh3
for path in "$models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors" \
    "$models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    "$models/vae/minimax_h3_image_vae_fp16.safetensors" "$stage/src/vendor" "$prompt"; do
    [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
mkdir -p "$out"
export PATH=$stage/rustup/toolchains/1.98.0-aarch64-unknown-linux-gnu/bin:$PATH CARGO_HOME=/work/cargo \
    CARGO_TARGET_DIR=/work/mmh3-target CUDA_HOME=/usr/local/cuda MMH3_CUDA_ARCH=sm_120f \
    LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
(cd "$stage/src" && cargo build --release --offline --locked --features cuda --bin mmh3) >"$out/build.log" 2>&1 ||
    { tail -40 "$out/build.log" >&2; exit 1; }
for side in 2048 4096; do
    log=$out/$side.log
    start=$(date +%s.%N)
    # TensorFold's prompt stage drops the file's last line ending; $(...) drops it here, mmh3 would keep it
    /work/mmh3-target/release/mmh3 image --models "$models" --prompt "$(cat "$prompt")" --seed 42 \
        --width $side --height $side --out "$out/$side.png" >"$log" 2>&1 || { cat "$log" >&2; exit 1; }
    echo "$start $(date +%s.%N)" | awk '{ printf "wall %.1f s\n", $2 - $1 }' >>"$log"
    echo "$side done"
done
