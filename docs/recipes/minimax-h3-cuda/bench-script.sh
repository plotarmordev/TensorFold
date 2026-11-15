#!/bin/sh
# run-script.sh's pictures as a benchmark, through the Desk's GPU door on TOP's Spark:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-script.sh"
# Same inputs and prebuilt bin-SHA/ tools. The selftests' PASS or FAIL is recorded, not a gate. Each size renders one
# cold picture and three warm repeats, every stage in its own process as in run-script.sh, while nvidia-smi samples
# memory and power. /work/out/h3-image-bench gets the pictures, the logs and SUMMARY.md (tools/zig/h3_image_bench.py).
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=$pilot/bin-$commit
out=/work/out/h3-image-bench
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
for path in "$model/model_index.json" "$model/text_encoder/config.json" "$pilot/prompt.txt" "$bin/tf-h3-image-cuda" "$bin/tf-h3-image-prompt-cuda" "$bin/tf-h3-image-decoder-cuda"; do
    [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
rm -rf "$out"
mkdir -p "$out"
{
    echo "commit: $commit"
    echo "GPU, driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
    echo "prompt: $pilot/prompt.txt, sha256 $(sha256sum <"$pilot/prompt.txt" | cut -d' ' -f1)"
    echo "seed: 42"
    echo "runs per size: 1 cold, 3 warm"
} >"$out/settings"
selftest() { # selftest NAME TOOL ARGS...: the log to selftest-NAME.log, PASS or FAIL to selftests
    name=$1
    shift
    if "$@" >"$out/selftest-$name.log" 2>&1; then echo "$name PASS"; else echo "$name FAIL"; fi >>"$out/selftests"
}
selftest transformer "$bin/tf-h3-image-cuda" --self-test
selftest decoder "$bin/tf-h3-image-decoder-cuda" selftest
selftest prompt "$bin/tf-h3-image-prompt-cuda" selftest /work/out/h3-prompt-selftest
sample() { # until killed, twice a second: time, GPU memory used (MiB), power (W), the compute processes' memory (MiB)
    while :; do
        echo "$(date +%s.%N), $(nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits)," \
            "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+ -)"
        sleep 0.5
    done
}
# the summary is written whenever the script exits, after a failed stage too
trap '[ -z "$sampler" ] || kill "$sampler"; python3 tools/zig/h3_image_bench.py "$out"' EXIT
stage() { # stage NAME TOOL ARGS...: the tool's log, with its times, to $case.NAME.log; shown if it fails
    log=$case.$1.log
    shift
    "$@" 2>"$log" || { cat "$log" >&2; exit 1; }
}
for side in 2048 4096; do
    for run in 1 2 3 4; do
        case=$out/$side-$run
        sample >"$case.smi" &
        sampler=$!
        date +%s.%N >"$case.time"
        stage prompt "$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$case.safetensors" "${side}x$side" 42
        stage image "$bin/tf-h3-image-cuda" "$model" "$case.safetensors" "$case"
        stage decoder "$bin/tf-h3-image-decoder-cuda" "$model" "$case.latent.f32" $((side / 16)) $((side / 16)) "$case.png" "${side}x$side"
        date +%s.%N >>"$case.time"
        kill "$sampler"
        sampler=
        echo "$side run $run done"
    done
done
