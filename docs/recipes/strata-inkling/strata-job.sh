#!/bin/sh
# Strata's own Inkling 2 Mini engine on TOP's GB10, set up as Strata publishes it, for card 242's receipt:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/strata-inkling/strata-job.sh
# SQ-20 and SQ-10, each under the engine's own `--bench` and, behind Strata's server, under our HTTP bench
# (tools/bench_concurrent.py), at 1, 4 and 64 streams, medians of 3. The receipt counts only if SQ-10 at 64 lands
# within 10% of the 805.8 tok/s of Strata's pack card; the last line says which.
# The job has no network. The Desk stages Strata v0.2.19 in the pilot folder strata-inkling-sq20/: the
# strata-inkling 0.1.65 and strata-control 0.1.65 linux-aarch64 packages and both packs (card 209's
# strata-inkling-pack/ is read for whatever is not there). The engine needs glibc 2.38 and GLIBCXX 3.4.32, newer than
# the job image's, so it runs under the loader, glibc and libstdc++ of Arm's GNU 14.3 toolchain staged for cuda-build.
set -eu
pilot=/assets/drama3-pilot
out=/work/out/strata-inkling
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
toolchain=$pilot/cuda-build-toolchain/$gcc.tar.xz
published=805.8
staged() { find "$pilot/strata-inkling-sq20" "$pilot/strata-inkling-pack" "$@" 2>/dev/null | head -1; }
engine=$(staged -name strata-inkling -type f)
server=$(staged -path '*/serve/server.py')
sq20=$(staged -name Inkling-2-Mini-SQ-20.strata)
sq10=$(staged -name Inkling-2-Mini-SQ-10.strata)
ls -l "$pilot/strata-inkling-sq20" || :
for need in "engine strata-inkling:$engine" "SQ-20 pack:$sq20" "SQ-10 pack:$sq10"; do
    [ -n "${need#*:}" ] || { echo "missing ${need%%:*} under $pilot/strata-inkling-sq20" >&2; exit 1; }
done
# the packs' digests as Strata pins them (terminal/strata_control/inkling_family.py at v0.2.19)
sha256sum -c --quiet <<EOF
0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97  $sq20
ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934  $sq10
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $toolchain
EOF
tar -C /work -xJf "$toolchain"
loader=$(find "/work/$gcc" -name ld-linux-aarch64.so.1 | head -1)
libc=$(dirname "$(find "/work/$gcc" -name libc.so.6 | head -1)")
libstdcxx=$(dirname "$(find "/work/$gcc" -name libstdc++.so.6 | head -1)")
strata=/work/strata-inkling # Strata's server knows the engine by this file name
printf '#!/bin/sh\nexec "%s" --library-path "%s" "%s" "$@"\n' \
    "$loader" "$(dirname "$engine"):$libc:$libstdcxx:/usr/lib/aarch64-linux-gnu" "$engine" >"$strata"
chmod +x "$strata"
mkdir -p "$out"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
version=$("$strata" --version)
echo "$version"
[ "$version" = "strata-inkling 0.1.65" ] || { echo "not Strata v0.2.19's engine: $engine" >&2; exit 1; }
cat <<'EOF'
[settings] Strata v0.2.19 is strata-inkling 0.1.65 (https://github.com/Niko1221/Strata/releases/tag/v0.2.19); its
  published bench flags, from the engine's --help (engine/src/inkling2/ink_main.cpp at that tag):
  "--bench --native PATH [--parallel 1..64] [--gpus 1|2] [--ep-split F] [--vram-reserve-mib N] [--cpu-threads N]
   [--kv auto|fp16|fp8|k8v8|k8v4] [--prompt-len N] [--gen N]"
  The pack card (https://huggingface.co/strata-engine/Inkling-2-Mini-Strata, "How speeds were measured") gives
  "| DGX Spark GB10 128 GB | SQ-20 / SQ-10 | 2,197 / 2,070 | 46.1 / 74.3 | 1,145.1 / 805.8 with 64 |" (prompt
  speed, one request, many at once), "Measured with thinking off", many at once being "the total written across
  that many requests running together"; it names neither the engine version, the flags nor the context.
[settings] own bench: --bench --native PACK --parallel N --gen 256, nothing else set.
  KV dtype: --kv auto, fp16 below 16 slots and fp8 from 16 (each [bench] line prints the one used).
  --parallel N: N slots, every slot decodes one token a step in one batched forward.
  prompt: 64 synthetic token ids a slot (--prompt-len's default), id 100 + (31*slot + position) % 1000: the same
  ramp shifted 31 ids a slot, not text and not one identical prompt. output: 256 greedy tokens a slot.
  drafting: DSpark on (the default); its tree runs at 1 slot only. graphs: the engine's default, no flag set.
  warmup: none, each run is one fresh process whose only pass is timed. 3 runs a cell, the median counts.
[settings] HTTP bench: Strata's server (strata-control 0.1.65, serve/server.py) with the run config its setup
  writes ("parallel": N slots, "drafting": "dspark", --max-context 32768, its default), under
  tools/bench_concurrent.py: N requests at once, 256 tokens, temperature 0, ignore_eos, --reps 3 (the tool's
  medians), one 16-token warm-up request a cell. Two cells a level: "same" sends tools/bench_prompts.json's first
  prompt on every stream, "different" sends request i that file's prompt i % 16 (at 64, each prompt on 4 streams).
EOF

bench() { # bench CELL PACK STREAMS [FLAG...]: one --bench run, its [bench] line to stdout and $out/CELL.txt
    cell=$1 pack=$2 streams=$3
    shift 3
    "$strata" --bench --native "$pack" --parallel "$streams" --gen 256 "$@" >"$out/bench.log" 2>&1 &&
        grep '^\[bench\]' "$out/bench.log" >>"$out/$cell.txt" || { tail -20 "$out/bench.log"; exit 1; }
    tail -1 "$out/$cell.txt"
}
median() { sed 's/.*decode=\([0-9.]*\) tps.*/\1/' "$out/$1.txt" | sort -n | sed -n 2p; }
for name in sq20 sq10; do
    eval "pack=\$$name"
    for streams in 1 4 64; do
        for rep in 1 2 3; do bench "$name-$streams" "$pack" "$streams"; done
        echo "[own] $name at $streams streams: median $(median "$name-$streams") tok/s total"
    done
done
# what could still differ from the published SQ-10 row at 64, one run each
bench probe "$sq10" 64 --kv fp16
bench probe "$sq10" 64 --draft off
bench probe "$sq10" 64 --gen 32

awk -v got="$(median sq10-64)" -v want="$published" 'BEGIN {
    ratio = got / want
    printf "[gate] SQ-10 at 64 streams, own bench: %.1f tok/s, %.0f%% of the published %.1f: %s\n", got, 100 * ratio,
        want, (ratio >= 0.9 && ratio <= 1.1) ? "counted" : "not counted, nothing in this receipt is a bar"
}'

[ -n "$server" ] || {
    echo "[http] not run: strata-control 0.1.65 (serve/server.py) is not staged under $pilot/strata-inkling-sq20"
    exit 1
}
control=$(dirname "$(dirname "$server")")
python=$control/python/bin/python3
python3 -c 'import json, sys; json.dump(json.load(open("tools/bench_prompts.json"))[:1], sys.stdout)' >"$out/same.json"
for name in sq20 sq10; do
    eval "pack=\$$name"
    "$python" "$control/tools/inkling_tokenizer.py" --strata "$pack" --out "$out/tokenizer-$name"
    for streams in 1 4 64; do
        cat >"$out/$name.json" <<EOF
{"exe": "$strata", "args": ["--native", "$pack", "--max-context", "32768"], "cwd": "/work", "arch": "inkling2_loop",
 "tokenizer": "$out/tokenizer-$name", "model_name": "$name", "drafting": "dspark", "parallel": $streams}
EOF
        PYTHONPATH=$control:$control/tools "$python" "$server" --engine strata --config "$out/$name.json" \
            --port 8242 --no-api-key >"$out/server-$name-$streams.log" 2>&1 &
        pid=$!
        until python3 -c 'import urllib.request as u; u.urlopen("http://127.0.0.1:8242/v1/models")' 2>/dev/null; do
            kill -0 "$pid" 2>/dev/null || { tail -30 "$out/server-$name-$streams.log"; exit 1; }
            sleep 2
        done
        for prompts in "$out/same.json" tools/bench_prompts.json; do
            python3 tools/bench_concurrent.py http://127.0.0.1:8242 "$name" --levels "$streams" --tokens 256 \
                --reps 3 --temperatures 0 --prompts "$prompts" --label "$name $(basename "$prompts" .json)" \
                --output "$out/http-$name-$streams-$(basename "$prompts")"
        done
        kill "$pid"
        wait "$pid" || :
        grep -h -o 'kv_dtype=[a-z0-9]*' "$out/server-$name-$streams.log" | sort -u
    done
done
