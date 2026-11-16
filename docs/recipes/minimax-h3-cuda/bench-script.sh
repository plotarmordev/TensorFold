#!/bin/sh
# run-script.sh's pictures as a benchmark through the Desk's GPU door; usage and output in ../minimax-h3-cuda.md.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=/work/out/h3-image-bench
if ! sh docs/recipes/minimax-h3-cuda/cuda-build.sh; then
    # The separate limits audit may fail; a build or copy failure has not reached its log.
    [ -f /work/out/limits.log ] || exit 1
fi
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
    echo "runs per size: 1 cold, 3 warm on one resident transformer (tf-h3-image-cuda --serve)"
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
sampler= server=
# the summary is written whenever the script exits, after a failed stage too
trap '[ -z "$sampler" ] || kill "$sampler"; [ -z "$server" ] || kill "$server" 2>/dev/null || :; python3 tools/zig/h3_image_bench.py "$out"' EXIT
stage() { # stage NAME TOOL ARGS...: the tool's log, with its times, to $case.NAME.log; shown if it fails
    log=$case.$1.log
    shift
    "$@" 2>"$log" || { cat "$log" >&2; exit 1; }
}
image() { # the resident transformer renders $case; the server's log since the last reply goes to $case.image.log
    [ -n "$server" ] || {
        mkfifo "$out/requests" "$out/replies"
        "$bin/tf-h3-image-cuda" "$model" --serve <"$out/requests" >"$out/replies" 2>"$out/$side.server.log" &
        server=$!
        exec 3>"$out/requests" 4<"$out/replies"
        logged=0
    }
    echo "$case.safetensors $case" >&3
    read -r verdict _ <&4 || verdict=
    tail -c +$((logged + 1)) "$out/$side.server.log" >"$case.image.log"
    logged=$(wc -c <"$out/$side.server.log")
    [ "$verdict" = done ] || { cat "$case.image.log" >&2; exit 1; }
}
for side in 2048 4096; do
    for run in 1 2 3 4; do
        case=$out/$side-$run
        sample >"$case.smi" &
        sampler=$!
        date +%s.%N >"$case.time"
        stage prompt "$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$case.safetensors" "${side}x$side" 42
        image
        stage decoder "$bin/tf-h3-image-decoder-cuda" "$model" "$case.latent.f32" $((side / 16)) $((side / 16)) "$case.png" "${side}x$side"
        date +%s.%N >>"$case.time"
        kill "$sampler"
        sampler=
        echo "$side run $run done"
    done
    # end of requests: the server exits; the next size loads its own, cold
    exec 3>&- 4<&-
    wait "$server"
    server=
    rm "$out/requests" "$out/replies"
    if [ "$side" = 2048 ]; then
        # the GPU decoder's 2048² picture beside the host reference decoder's on the same latent
        case=$out/2048-reference
        stage compare "$bin/tf-h3-image-decoder-cuda" reference "$model" "$out/2048-1.latent.f32" 128 128 \
            "$out/2048-reference-gpu.png" "$out/2048-reference-host.png" 2048x2048
    fi
done
