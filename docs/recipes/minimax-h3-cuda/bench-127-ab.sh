#!/bin/sh
# Card 127's same-job A/B: the resident prompt server keeps a prompt's states, so a repeated prompt
# skips the encoder and refiner (prompt ~1.1 s -> ~0) while an unseen prompt pays the same encode
# as main. Runs on this commit, then bench-127-base.patch is reverse-applied to put the card's
# files back to main's and the same runs repeat (the job clone is too shallow for a git revert).
# Each phase: the selftests, then one resident prompt/transformer/decoder server takes 8 requests —
# the pilot's prompt cold then twice warm (each warm run a kept-states hit on this commit: repeated),
# then five unseen prompts — wall, prompt, transformer and decoder times per run plus the PNG of the
# last repeat compared byte-for-byte with the cold one's.
# Results: /work/out/h3-cache-ab/SUMMARY.txt has both phases' labelled numbers.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-127-ab.sh"
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
patch_file=docs/recipes/minimax-h3-cuda/bench-127-base.patch
out=/work/out/h3-cache-ab
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# the toolchain once, as cuda-build.sh stages it (the job has neither zig nor gcc in PATH)
staged=/assets/drama3-pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
[ -d "$staged" ] || { echo "missing $staged: run toolchain-stage.sh" >&2; exit 1; }
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
[ -d "/work/$zig" ] || tar -C /work -xJf "$staged/$zig.tar.xz"
[ -d "/work/$gcc" ] || tar -C /work -xJf "$staged/$gcc.tar.xz"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache \
    NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"

mkdir -p "$out" "$bin"
# three prompts the server has not seen, same files in both phases: the pilot's prompt with a
# detail appended, one file each (the three unseen runs the card asks for)
for n in 1 2 3; do { cat "$pilot/prompt.txt"; printf ' — variant %s of the same scene\n' "$n"; } >"$out/prompt-$n.txt"; done

# phase NAME: build the three picture tools into $bin, selftest them, run 8 pictures through the
# resident servers, write per-run times to $dir/times
phase() {
    name=$1
    dir=$out/$name
    mkdir -p "$dir"
    zig build tf-h3-image-prompt-cuda tf-h3-image-cuda tf-h3-image-decoder-cuda \
        -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix "/work/sm121-$name" \
        >"$dir/build.log" 2>&1 || { tail -20 "$dir/build.log" >&2; exit 1; }
    cp "/work/sm121-$name/bin/tf-h3-image-prompt-cuda" "/work/sm121-$name/bin/tf-h3-image-cuda" \
        "/work/sm121-$name/bin/tf-h3-image-decoder-cuda" "$bin"
    selftest() {
        tname=$1
        shift
        if "$@" >"$dir/selftest-$tname.log" 2>&1; then echo "$tname PASS" >>"$dir/selftests"; else
            echo "$tname FAIL" >>"$dir/selftests"
            tail -20 "$dir/selftest-$tname.log" >&2
            exit 1
        fi
    }
    selftest transformer "$bin/tf-h3-image-cuda" --self-test
    selftest decoder "$bin/tf-h3-image-decoder-cuda" selftest --tc
    selftest prompt "$bin/tf-h3-image-prompt-cuda" selftest "$dir/prompt-selftest"

    mkfifo "$dir/requests" "$dir/replies" "$dir/prompt-requests" "$dir/prompt-replies" "$dir/decoder-requests" "$dir/decoder-replies"
    "$bin/tf-h3-image-cuda" "$model" --serve <"$dir/requests" >"$dir/replies" 2>"$dir/server.log" \
        3>&- 4>&- 5>&- 6>&- 7>&- 8>&- &
    server=$!
    "$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" --serve <"$dir/prompt-requests" >"$dir/prompt-replies" 2>"$dir/prompt-server.log" \
        3>&- 4>&- 5>&- 6>&- 7>&- 8>&- &
    prompt_server=$!
    "$bin/tf-h3-image-decoder-cuda" "$model" --serve --tc <"$dir/decoder-requests" >"$dir/decoder-replies" 2>"$dir/decoder-server.log" \
        3>&- 4>&- 5>&- 6>&- 7>&- 8>&- &
    decoder_server=$!
    exec 3>"$dir/requests" 4<"$dir/replies" 5>"$dir/prompt-requests" 6<"$dir/prompt-replies" \
        7>"$dir/decoder-requests" 8<"$dir/decoder-replies"
    logged=0
    prompt_logged=0
    decoder_logged=0
    run() { # run LABEL PROMPT_FILE: one 2048x2048 seed-42 picture end to end; its stage seconds to $dir/times
        label=$1
        prompt_file=$2
        case=$dir/$label
        start=$(date +%s.%N)
        echo "$prompt_file $case.safetensors 2048x2048 42" >&5
        read -r verdict _ <&6
        [ "$verdict" = done ] || { tail -10 "$dir/prompt-server.log" >&2; exit 1; }
        tail -c +$((prompt_logged + 1)) "$dir/prompt-server.log" >"$case.prompt.log"
        prompt_logged=$(wc -c <"$dir/prompt-server.log")
        echo "$case.safetensors $case" >&3
        read -r verdict _ <&4
        [ "$verdict" = done ] || { tail -10 "$dir/server.log" >&2; exit 1; }
        tail -c +$((logged + 1)) "$dir/server.log" >"$case.image.log"
        logged=$(wc -c <"$dir/server.log")
        echo "$case.latent.f32 128 128 $case.png 2048x2048" >&7
        read -r verdict _ <&8
        [ "$verdict" = done ] || { tail -10 "$dir/decoder-server.log" >&2; exit 1; }
        tail -c +$((decoder_logged + 1)) "$dir/decoder-server.log" >"$case.decoder.log"
        decoder_logged=$(wc -c <"$dir/decoder-server.log")
        end=$(date +%s.%N)
        python3 - "$label" "$start" "$end" "$case" >>"$dir/times" <<'PY'
import os, re, sys
label, start, end, case = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
def found(log, pattern):
    m = re.search(pattern, open(log, errors="replace").read(), re.M) if os.path.exists(log) else None
    return float(m.group(1)) if m else float("nan")
prompt = found(case + ".prompt.log", r"^prompt: ([\d.]+) s$")
transformer = found(case + ".image.log", r"^transformer: ([\d.]+) s$")
decoder = found(case + ".decoder.log", r"([\d.]+) s in all;")
print(f"{label} wall {end - start:.2f} prompt {prompt:.2f} transformer {transformer:.2f} decoder {decoder:.2f}")
PY
        echo "$name $label done"
    }

    : >"$dir/times"
    run cold "$pilot/prompt.txt"
    run repeat-1 "$pilot/prompt.txt"
    run repeat-2 "$pilot/prompt.txt"
    run repeat-3 "$pilot/prompt.txt"
    for n in 1 2 3; do run "unseen-$n" "$out/prompt-$n.txt"; done
    exec 3>&- 4<&- 5>&- 6<&- 7>&- 8<&-
    wait "$server"
    wait "$prompt_server"
    wait "$decoder_server"

    # the repeated prompt's picture must be byte-identical to the cold run's (same prompt, seed, size)
    cmp "$dir/cold.png" "$dir/repeat-3.png" >/dev/null \
        && echo "$name: repeat-3.png byte-identical to cold.png" >>"$dir/times" \
        || { echo "$name: repeat-3.png DIFFERS from cold.png" >>"$dir/times"; exit 1; }
}

echo "=== this commit: prompt states kept ==="
phase candidate
echo "=== main: the prompt re-encoded every picture ==="
git -c safe.directory='*' apply -R "$patch_file"
status=0
phase baseline || status=$?
git -c safe.directory='*' apply "$patch_file"

{
    echo "repeated prompt = a warm run of the pilot's prompt in one resident server (a kept-states hit on this"
    echo "commit, a re-encode on main); unseen prompt = a prompt the server had not seen (a miss either way)."
    for name in candidate baseline; do
        echo "--- $name ---"
        cat "$out/$name/times"
    done
} | tee "$out/SUMMARY.txt"
exit "$status"
