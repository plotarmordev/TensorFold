#!/bin/sh
# Card 145's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-145-base.patch reverse-applied — the card's files back to main's — and the same
# unseen-prompt runs again: the baseline is the same job's main code with this commit's bench script,
# model paths and --tc decoder default. Each phase renders three pictures the server has not seen
# (the pilot's prompt with a variant line appended), under H3_PROFILE so the transformer's stage
# times land in the logs; SUMMARY.txt lists every run's wall, prompt, transformer, decoder and the
# H3_PROFILE block stages.
# Results: /work/out/h3-145-ab (this commit) and its baseline/ subfolder (the card's files at main).
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-145-ab.sh"
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
patch_file=docs/recipes/minimax-h3-cuda/bench-145-base.patch
out=/work/out/h3-145-ab
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
    NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++" H3_PROFILE=1

mkdir -p "$out"
rm -f "$out/limits-failed"
# three prompts the server has not seen, the same files in both phases, after a warm-up (variant 0) each phase
for n in 0 1 2 3; do { cat "$pilot/prompt.txt"; printf ' — variant %s of the same scene\n' "$n"; } >"$out/prompt-$n.txt"; done

# phase NAME: build the three picture tools, selftest them, run the three unseen prompts through the
# resident servers, write every run's stage seconds to $dir/times
phase() {
    name=$1
    dir=$out/$name
    bin=/work/out/tf145-$name
    mkdir -p "$dir" "$bin"
    zig build tf-h3-image-prompt-cuda tf-h3-image-cuda tf-h3-image-decoder-cuda \
        -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix "/work/sm121-$name" \
        >"$dir/build.log" 2>&1 || { tail -20 "$dir/build.log" >&2; exit 1; }
    cp "/work/sm121-$name/bin/tf-h3-image-prompt-cuda" "/work/sm121-$name/bin/tf-h3-image-cuda" \
        "/work/sm121-$name/bin/tf-h3-image-decoder-cuda" "$bin"
    # a five-image fatbin build once a phase, so the h3 records' sm_80/86/89/120/121 numbers can be
    # re-derived from the kernels this phase compiled (registry_media.zig's usage columns)
    zig build tf-cuda-limits -Dsm=80,86,89,120,121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast \
        --prefix "/work/sm-all-$name" >"$dir/limits-build.log" 2>&1 || { tail -20 "$dir/limits-build.log" >&2; exit 1; }
    fatdir=$(find "/work/sm-all-$name" -name h3.fatbin -exec dirname {} \; -quit)
    # the audit's verdict fails the job at the end, after both phases' timings
    "/work/sm-all-$name/bin/tf-cuda-limits" "$fatdir" h3 >"$dir/limits.log" 2>&1 \
        || { echo "$name limits audit FAILED" | tee -a "$out/limits-failed" >&2; tail -20 "$dir/limits.log" >&2; }
    grep "RECORD h3" "$dir/limits.log" >"$dir/records.txt" || true
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
for stage in re.findall(r"^  ([a-z ]+): ([\d.]+) s$", open(case + ".image.log", errors="replace").read(), re.M):
    print(f"    {stage[0]}: {stage[1]} s")
PY
        sha256sum "$case.latent.f32" | cut -c1-16 >>"$dir/times"
        echo "$name $label done"

    }

    : >"$dir/times"
    run warmup "$out/prompt-0.txt"
    for n in 1 2 3; do run "unseen-$n" "$out/prompt-$n.txt"; done
    exec 3>&- 4<&- 5>&- 6<&- 7>&- 8<&-
    wait "$server"
    wait "$prompt_server"
    wait "$decoder_server"
}

echo "=== this commit: the fused int8 passes ==="
phase candidate
echo "=== main: the card's files at main ==="
git -c safe.directory='*' apply -R "$patch_file"
status=0
(phase baseline) || status=$?
git -c safe.directory='*' apply "$patch_file"
[ ! -e "$out/limits-failed" ] || status=1

# every latent must be byte-identical between the phases: the card's products and rounding do not change
for n in 1 2 3; do
    cmp -s "$out/candidate/unseen-$n.latent.f32" "$out/baseline/unseen-$n.latent.f32" \
        && echo "unseen-$n latent byte-identical" >>"$out/latents.txt" \
        || { echo "unseen-$n latent DIFFERS" >>"$out/latents.txt"; status=1; }
done

{
    echo "unseen prompt = a prompt the resident server had not seen; three warm runs a phase, the same"
    echo "prompt files both sides; latents' sha256 prefixes beside each run."
    for name in candidate baseline; do
        echo "--- $name ---"
        cat "$out/$name/times"
    done
} | tee "$out/SUMMARY.txt"
cat "$out/latents.txt" | tee -a "$out/SUMMARY.txt"
exit "$status"
