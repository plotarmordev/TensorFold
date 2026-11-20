#!/bin/sh
# Card 142's same-job A/B for the threaded transformer load: --load-threads 0 (today's in-place
# reads) against --load-threads 8 (pread readers into the pinned stages). One build, both counts the
# same binary. Each cold run evicts every weight file's pages with posix_fadvise(DONTNEED), runs the
# prompt case through one-shot tf-h3-image-cuda (its "model loaded in" line is the transformer
# load), then a resident decoder writes the PNG. Three cold runs a thread count, alternating
# 0/8/0/8/0/8 so drift can't favour one; the warm picture is a resident transformer's second
# render, once a count. Every latent and PNG is compared byte-for-byte against the 0-thread run.
# Results: /work/out/h3-load-ab/SUMMARY.txt.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-142-ab.sh
set -eu

# summary DIR THREADS: every threaded run's outputs against the 0-thread run's bytes, then the
# times; fails when any output differs
summary() {
    status=0
    for run in 1 2 3; do
        for ext in latent.f32 png; do
            if cmp -s "$1/cold-$run-t0.$ext" "$1/cold-$run-t$2.$ext"; then
                echo "cold-$run $ext: byte-identical"
            else
                echo "cold-$run $ext: DIFFERS"
                status=1
            fi
        done
    done
    for ext in latent.f32 png; do
        if cmp -s "$1/warm-t0-2.$ext" "$1/warm-t$2-2.$ext"; then
            echo "warm-2 $ext: byte-identical"
        else
            echo "warm-2 $ext: DIFFERS"
            status=1
        fi
    done
    echo
    cat "$1/times"
    return "$status"
}
if [ "${1:-}" = summary ]; then
    summary "$2" "$3"
    exit
fi

# before any GPU work: the script's own summary must fail on one planted differing PNG
probe=$(mktemp -d)
for name in cold-1-t0 cold-1-t8 cold-2-t0 cold-2-t8 cold-3-t0 cold-3-t8 warm-t0-2 warm-t8-2; do
    printf same >"$probe/$name.latent.f32"
    printf same >"$probe/$name.png"
done
printf other >"$probe/cold-2-t8.png"
: >"$probe/times"
if sh docs/recipes/minimax-h3-cuda/bench-142-ab.sh summary "$probe" 8 >/dev/null; then
    echo "the summary passed a differing PNG" >&2
    exit 1
fi
rm -rf "$probe"

model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=/work/out/h3-load-ab
threads=${H3_LOAD_THREADS:-8}
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
rm -rf "$out"
mkdir -p "$out"
for path in "$model/model_index.json" "$pilot/prompt.txt" "$bin/tf-h3-image-cuda" \
    "$bin/tf-h3-image-prompt-cuda" "$bin/tf-h3-image-decoder-cuda"; do
    [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
echo "commit: $commit" | tee "$out/settings"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$out/settings"
echo "prompt: $pilot/prompt.txt seed 42, 2048x2048, six steps" | tee -a "$out/settings"
echo "load threads: 0 (in place) vs $threads (pread readers)" | tee -a "$out/settings"

selftest() { # selftest NAME TOOL ARGS...: PASS to selftests, the log to selftest-NAME.log
    name=$1; shift
    if "$@" >"$out/selftest-$name.log" 2>&1; then echo "$name PASS" >>"$out/selftests"; else
        echo "$name FAIL" >>"$out/selftests"; tail -20 "$out/selftest-$name.log" >&2; exit 1
    fi
}
selftest transformer "$bin/tf-h3-image-cuda" --self-test
selftest decoder "$bin/tf-h3-image-decoder-cuda" selftest --tc

# the picture case once: identical bytes for every run (same prompt, size, seed)
"$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" \
    "$out/case.safetensors" 2048x2048 42 >"$out/case.log" 2>&1 || { tail -20 "$out/case.log" >&2; exit 1; }

# drop every model weight file's cached pages so the next load reads storage cold; the resident
# transformer's own evict() proves the mechanism on every picture it serves
evict() {
    find "$model" -name '*.safetensors' -print | python3 -c '
import os, sys
for path in sys.stdin:
    fd = os.open(path.strip(), os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)'
}

decoder_server=
decoder_stop() {
    [ -z "$decoder_server" ] || {
        exec 7>&- 8<&-
        wait "$decoder_server" 2>/dev/null || :
        decoder_server=
        rm -f "$out/decoder-requests" "$out/decoder-replies"
    }
}
decoder_start() {
    mkfifo "$out/decoder-requests" "$out/decoder-replies"
    "$bin/tf-h3-image-decoder-cuda" "$model" --serve --tc <"$out/decoder-requests" \
        >"$out/decoder-replies" 2>"$out/$1.decoder-server.log" 3>&- 4>&- 5>&- 6>&- &
    decoder_server=$!
    exec 7>"$out/decoder-requests" 8<"$out/decoder-replies"
}
decode() { # decode LABEL OUT_PREFIX: one PNG through the resident decoder
    echo "$2.latent.f32 128 128 $2.png 2048x2048" >&7
    read -r verdict _ <&8 || verdict=
    [ "$verdict" = done ] || { tail -10 "$out/main.decoder-server.log" >&2; exit 1; }
}
trap 'decoder_stop' EXIT


# The transformer load needs its own evicted run per measurement (a resident server loads before
# any request), so each timing is a one-shot tool, alternating thread counts run to run:
#   A0 = --load-threads 0, B8 = --load-threads 8, three runs each.
run_cold() {
    run=$1
    count=$2
    case=$out/cold-$run-t$count
    evict
    start=$(date +%s.%N)
    "$bin/tf-h3-image-cuda" "$model" "$out/case.safetensors" "$case" --load-threads "$count" \
        >"$case.image.log" 2>&1 || { cat "$case.image.log" >&2; exit 1; }
    decode "cold-$run-t$count" "$case"
    end=$(date +%s.%N)
    python3 - "$run" "$count" "$start" "$end" "$case.image.log" >>"$out/times" <<'PY'
import re, sys
run, threads, start, end, log = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:5]), sys.argv[5]
text = open(log, errors="replace").read()
load = float(re.search(r"model loaded in ([\d.]+) s", text).group(1))
transformer = float(re.search(r"^transformer: ([\d.]+) s$", text, re.M).group(1))
print(f"cold-{run} threads={threads} wall {end - start:.2f} load {load:.2f} transformer {transformer:.2f}")
PY
    echo "cold-$run threads=$count done"
}

decoder_start main
: >"$out/times"
# alternating runs, three each: 0 then the readers, so storage drift hits both phases the same
for run in 1 2 3; do
    run_cold "$run" 0
    run_cold "$run" "$threads"
done

# warm pictures on the resident transformer, once per thread count (the load path cannot change
# a step's math): the server stays up between its two renders
for count in 0 "$threads"; do
    mkfifo "$out/requests" "$out/replies"
    "$bin/tf-h3-image-cuda" "$model" --serve --load-threads "$count" <"$out/requests" \
        >"$out/replies" 2>"$out/warm-t$count.server.log" 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- &
    server=$!
    exec 3>"$out/requests" 4<"$out/replies"
    logged=0
    # first render finishes the load; the second is the warm picture
    for run in 1 2; do
        case=$out/warm-t$count-$run
        start=$(date +%s.%N)
        echo "$out/case.safetensors $case" >&3
        read -r verdict _ <&4 || verdict=
        [ "$verdict" = done ] || { tail -10 "$out/warm-t$count.server.log" >&2; exit 1; }
        decode "warm-t$count-$run" "$case"
        end=$(date +%s.%N)
        tail -c +$((logged + 1)) "$out/warm-t$count.server.log" >"$case.image.log" 2>/dev/null || :
        logged=$(wc -c <"$out/warm-t$count.server.log")
        python3 - "$run" "$count" "$start" "$end" "$case.image.log" >>"$out/times" <<'PY'
import re, sys
run, threads, start, end, log = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:5]), sys.argv[5]
text = open(log, errors="replace").read()
m = re.search(r"^transformer: ([\d.]+) s$", text, re.M)
print(f"warm-{run} threads={threads} wall {end - start:.2f} transformer {float(m.group(1)):.2f}")
PY
    done
    exec 3>&- 4<&-
    wait "$server"
    rm "$out/requests" "$out/replies"
done
decoder_stop
trap - EXIT

# every threaded run's outputs must be the 0-thread run's bytes
status=0
summary "$out" "$threads" >"$out/SUMMARY.txt" || status=1
cat "$out/SUMMARY.txt"
exit "$status"
