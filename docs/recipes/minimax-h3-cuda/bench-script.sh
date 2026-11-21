#!/bin/sh
# run-script.sh's pictures as a benchmark through the Desk's GPU door; usage and output in ../minimax-h3-cuda.md.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=${H3_BENCH_OUT:-/work/out/h3-image-bench}
# the sm_121 tools only: the five SMs' build and limits audit are cuda-build.sh's own job
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
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
    echo "sizes: ${H3_BENCH_SIZES:-2048}"
    echo "transformer linear layers: ${H3_LINEAR:-int8}"
    echo "draft steps: ${H3_DRAFT_STEPS:-0}${H3_RENOISE_SIGMA:+, renoise sigma $H3_RENOISE_SIGMA}"
    echo "prompts per run: ${H3_BENCH_PROMPTS:-$pilot/prompt.txt (all runs, each a kept-states hit)}"
} >"$out/settings"
selftest() { # selftest NAME TOOL ARGS...: the log to selftest-NAME.log, PASS or FAIL to selftests
    name=$1
    shift
    if "$@" >"$out/selftest-$name.log" 2>&1; then echo "$name PASS"; else echo "$name FAIL"; fi >>"$out/selftests"
}
selftest transformer "$bin/tf-h3-image-cuda" --self-test
case ${H3_DECODER:-} in exact) decflag=--exact ;; tc) decflag=--tc ;; *) decflag= ;; esac
case ${H3_LINEAR:-int8} in bf16) linflag=--bf16 ;; fp4) linflag=--fp4 ;; *) linflag= ;; esac
selftest decoder "$bin/tf-h3-image-decoder-cuda" selftest $decflag
selftest prompt "$bin/tf-h3-image-prompt-cuda" selftest /work/out/h3-prompt-selftest
sample() { # until killed, twice a second: time, GPU memory used (MiB), power (W), the compute processes' memory (MiB)
    while :; do
        echo "$(date +%s.%N), $(nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits)," \
            "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | paste -sd+ -)"
        sleep 0.5
    done
}
sampler= server= prompt_server= decoder_server=
# the summary is written whenever the script exits, after a failed stage too
trap '[ -z "$sampler" ] || kill "$sampler"; [ -z "$server" ] || kill "$server" 2>/dev/null || :; [ -z "$prompt_server" ] || kill "$prompt_server" 2>/dev/null || :; [ -z "$decoder_server" ] || kill "$decoder_server" 2>/dev/null || :; python3 tools/zig/h3_image_bench.py "$out"' EXIT
stage() { # stage NAME TOOL ARGS...: the tool's log, with its times, to $case.NAME.log; shown if it fails
    log=$case.$1.log
    shift
    "$@" 2>"$log" || { cat "$log" >&2; exit 1; }
}
image() { # the resident transformer renders $case; the server's log since the last reply goes to $case.image.log
    [ -n "$server" ] || {
        mkfifo "$out/requests" "$out/replies"
        # each server closes this script's FIFO fds: a later server would inherit an earlier one's write end and
        # the earlier server would never see EOF when the script closes it
        # H3_PROFILE (off unless set) waits after every stage for its block-stage lines; the default run
        # captures each step as one CUDA graph, which those waits would keep it from doing
        # H3_DRAFT_STEPS (+ optional H3_RENOISE_SIGMA) names --draft-steps (and --renoise-sigma): mmh3's
        # half-canvas drafts, beside bench-drafts.sh like H3_LINEAR sits beside bench-bf16.sh
        # shellcheck disable=SC2046 # $linflag is empty at the int8 default; --bf16 and --fp4 are the opt-ins
        H3_PROFILE=${H3_PROFILE-} "$bin/tf-h3-image-cuda" "$model" --serve $linflag ${H3_DRAFT_STEPS:+--draft-steps "$H3_DRAFT_STEPS"} ${H3_RENOISE_SIGMA:+--renoise-sigma "$H3_RENOISE_SIGMA"} <"$out/requests" >"$out/replies" 2>"$out/$side.server.log" \
            3>&- 4>&- 5>&- 6>&- 7>&- 8>&- &
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
prompt_stage() { # the resident prompt stage writes $case's picture case from $1; its new log bytes go to $case.prompt.log
    prompt_file=$1
    [ -n "$prompt_server" ] || {
        mkfifo "$out/prompt-requests" "$out/prompt-replies"
        "$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" --serve <"$out/prompt-requests" >"$out/prompt-replies" 2>"$out/$side.prompt-server.log" \
            3>&- 4>&- 5>&- 6>&- 7>&- 8>&- &
        prompt_server=$!
        exec 5>"$out/prompt-requests" 6<"$out/prompt-replies"
        prompt_logged=0
    }
    echo "$prompt_file $case.safetensors ${side}x$side 42" >&5
    read -r verdict _ <&6 || verdict=
    tail -c +$((prompt_logged + 1)) "$out/$side.prompt-server.log" >"$case.prompt.log"
    prompt_logged=$(wc -c <"$out/$side.prompt-server.log")
    [ "$verdict" = done ] || { cat "$case.prompt.log" >&2; exit 1; }
}
decoder_stage() { # the resident decoder writes $case.png; its new log bytes go to $case.decoder.log
    [ -n "$decoder_server" ] || {
        mkfifo "$out/decoder-requests" "$out/decoder-replies"
        # the decoder's tensor-core path is the default; H3_DECODER=exact passes --exact for fp32 order
        H3_PROFILE=${H3_PROFILE-1} "$bin/tf-h3-image-decoder-cuda" "$model" --serve $decflag <"$out/decoder-requests" >"$out/decoder-replies" 2>"$out/$side.decoder-server.log" \
            3>&- 4>&- 5>&- 6>&- 7>&- 8>&- &
        decoder_server=$!
        exec 7>"$out/decoder-requests" 8<"$out/decoder-replies"
        decoder_logged=0
    }
    echo "$case.latent.f32 $((side / 16)) $((side / 16)) $case.png ${side}x$side" >&7
    read -r verdict _ <&8 || verdict=
    tail -c +$((decoder_logged + 1)) "$out/$side.decoder-server.log" >"$case.decoder.log"
    decoder_logged=$(wc -c <"$out/$side.decoder-server.log")
    [ "$verdict" = done ] || { cat "$case.decoder.log" >&2; exit 1; }
}
# H3_BENCH_PROMPTS is one prompt file a run; run n takes the nth word, so a file listed twice is a
# kept-states hit and a file listed once an unseen prompt. Unset, every run re-reads the pilot prompt.
for side in ${H3_BENCH_SIZES:-2048}; do
    run=0
    for prompt_file in ${H3_BENCH_PROMPTS:-$pilot/prompt.txt $pilot/prompt.txt $pilot/prompt.txt $pilot/prompt.txt}; do
        run=$((run + 1))
        case=$out/$side-$run
        sample >"$case.smi" &
        sampler=$!
        date +%s.%N >"$case.time"
        prompt_stage "$prompt_file"
        image
        decoder_stage
        date +%s.%N >>"$case.time"
        kill "$sampler"
        sampler=
        echo "$side run $run done"
    done
    # end of requests: the servers exit; the next size loads its own, cold
    exec 3>&- 4<&-
    wait "$server"
    server=
    rm "$out/requests" "$out/replies"
    exec 5>&- 6<&-
    wait "$prompt_server"
    prompt_server=
    rm "$out/prompt-requests" "$out/prompt-replies"
    exec 7>&- 8<&-
    wait "$decoder_server"
    decoder_server=
    rm "$out/decoder-requests" "$out/decoder-replies"
done
# the GPU decoder's 2048² picture beside the host reference decoder's on the same latent, after the
# resident servers freed their arenas: the host decoder needs the memory they held. H3_BENCH_SKIP_COMPARE
# skips it for a same-job A/B's second phase, which needs no second host decode.
if [ -z "${H3_BENCH_SKIP_COMPARE:-}" ] && [ -f "$out/2048-1.latent.f32" ]; then
    case=$out/2048-reference
    stage compare "$bin/tf-h3-image-decoder-cuda" reference "$model" "$out/2048-1.latent.f32" 128 128 \
        "$out/2048-reference-gpu.png" "$out/2048-reference-host.png" 2048x2048 $decflag
fi
# a failed selftest still gets its timings and summary, then fails the job
if grep -q " FAIL$" "$out/selftests"; then exit 1; fi
