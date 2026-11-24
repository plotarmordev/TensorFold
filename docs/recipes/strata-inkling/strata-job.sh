#!/bin/sh
# Strata's own Inkling 2 Mini engine on TOP's GB10, set up as Strata publishes it, for card 247's receipt. One GPU
# job a pack and level, as the Desk queues them (job start passes no env or args):
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/strata-inkling/strata-sq20-1.sh
# and likewise strata-sq20-4.sh, strata-sq20-64.sh, strata-sq10-1.sh, strata-sq10-4.sh, strata-sq10-64.sh; they set
# STRATA_PACK (sq20 or sq10) and STRATA_STREAMS (1, 4 or 64). Unset, this script runs every pack and level.
# strata-http-sq20.sh and strata-http-sq10.sh set STRATA_OWN=0: one pack's HTTP bench at every level, no own bench.
# strata-own.sh sets STRATA_HTTP=0: both packs' own bench at every level in one job, no HTTP bench.
# Each pack, under the engine's own `--bench` and, behind Strata's server, under our HTTP bench
# (tools/bench_concurrent.py), medians of 3. The receipt counts only if SQ-10 at 64 lands within 10% of the
# 805.8 tok/s of Strata's pack card; the sq10-64 job's [gate] line says which.
# The Desk stages v0.2.19's engine, Control and tokenizer in strata-inkling-0219/; the SQ-20 and SQ-10 packs
# already live in strata-inkling-sq20/ and strata-inkling-pack/. The job has no network. The engine needs glibc 2.38
# and GLIBCXX 3.4.32, supplied by the staged Arm GNU 14.3 toolchain.
# Setup runs first and logs "build: N s"; each cell's measurement is one `timeout 600` phase ("measure HALF PACK
# STREAMS: N s"), so a one-job both-benches run stays inside the slot bound. STRATA_CELL=own|http runs one half.
set -eu
here=$(dirname "$0")
pilot=/assets/drama3-pilot
out=/work/out/strata-inkling
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
toolchain=$pilot/cuda-build-toolchain/$gcc.tar.xz
published=805.8
release=$pilot/strata-inkling-0219
engine=$(find "$release/inkling" -name strata-inkling -type f 2>/dev/null | head -1)
server=$(find "$release/control" -path '*/serve/server.py' -type f 2>/dev/null | head -1)
sq20=$pilot/strata-inkling-sq20/Inkling-2-Mini-SQ-20.strata
sq10=$pilot/strata-inkling-pack/Inkling-2-Mini-SQ-10.strata
python=$(find "$release/control" -path '*/python/bin/python3' -type f 2>/dev/null | head -1)
control=$(dirname "$(dirname "$server")")
strata=/work/strata-inkling # Strata's server knows the engine by this file name
packs=${STRATA_PACK:-sq20 sq10}
levels=${STRATA_STREAMS:-1 4 64}

bench() { # bench CELL PACK STREAMS [FLAG...]: one --bench run, its [bench] line to stdout and $out/CELL.txt
    cell=$1 pack=$2 streams=$3
    shift 3
    "$strata" --bench --native "$pack" --parallel "$streams" "$@" >"$out/bench.log" 2>&1 &&
        grep '^\[bench\]' "$out/bench.log" >>"$out/$cell.txt" || { tail -20 "$out/bench.log"; exit 1; }
    tail -1 "$out/$cell.txt"
}
median() { sed 's/.*decode=\([0-9.]*\) tps.*/\1/' "$out/$1.txt" | sort -n | sed -n 2p; }
# STRATA_CELL (own or http) limits a level wrapper to one half: a shared out/ keeps the other half's cells.
own() { [ -z "${STRATA_CELL:-}" ] || [ "$STRATA_CELL" = own ]; }
http() { [ -z "${STRATA_CELL:-}" ] || [ "$STRATA_CELL" = http ]; }
summary() { # summary REPORT: one [http] line a cell; fails on a failed request or a cell without a client-clock rate
    python3 - "$1" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
for cell in report["cells"]:
    asked = [r for rep in cell["reps"] for r in rep["evidence"]]
    prompt = [r["prompt_tokens"] or 0 for r in asked]
    tokens = [r["tokens"] or 0 for r in asked]
    print("[http] %s, %d streams: %s tok/s total end to end (usage tokens over the makespan, prefill included, "
          "median of 3); prompt %d-%d tokens, %d-%d of %d output tokens, completions %s, failed %d" % (
              report["label"], cell["streams"], "%.1f" % cell["end_to_end_tps"] if cell["timing_valid"] else "-",
              min(prompt), max(prompt), min(tokens), max(tokens), report["tokens"], cell["completions"],
              cell["failed"]))
    if cell["failed"] or not cell["timing_valid"]:
        sys.exit(1)
EOF
}
stop() { # the running Strata server, if any: on every exit, so a failed cell never leaves it serving
    [ -z "${pid:-}" ] || { kill "$pid" 2>/dev/null; wait "$pid" || :; }
    pid=
}
measure() { # every pack and level asked for, own bench (unless STRATA_OWN=0) then HTTP bench (unless STRATA_HTTP=0)
    trap stop EXIT
    trap 'exit 1' TERM INT
    gate=
    for name in $packs; do
        eval "pack=\$$name"
        for streams in $levels; do
            own && [ "${STRATA_OWN:-1}" = 1 ] || continue
            : >"$out/$name-$streams.txt"
            for rep in 1 2 3; do bench "$name-$streams" "$pack" "$streams"; done
            echo "[own] $name at $streams streams: median $(median "$name-$streams") tok/s total"
            [ "$name-$streams" != sq10-64 ] || gate=1
        done
    done
    # The pack card does not disclose engine version, flags, context or prompt content; the gate is empirical and
    # prints only for a sq10-64 cell this job measured: never a stale file shared out/ left from an earlier run.
    [ -z "$gate" ] || awk -v got="$(median sq10-64)" -v want="$published" 'BEGIN {
        ratio = got / want
        verdict = (ratio >= 0.9 && ratio <= 1.1) ? "counted" : "not counted, nothing in this receipt is a bar"
        printf "[gate] SQ-10 at 64 streams, own bench: %.1f tok/s, %.0f%% of the published %.1f: %s\n", got,
            100 * ratio, want, verdict
    }'
    http && [ "${STRATA_HTTP:-1}" = 1 ] || return 0
    for name in $packs; do
        eval "pack=\$$name"
        for streams in $levels; do
            cat >"$out/$name.json" <<EOF
{"exe": "$strata", "args": ["--native", "$pack", "--max-context", "32768"], "cwd": "/work", "arch": "inkling2_loop",
 "tokenizer": "$out/tokenizer-$name", "model_name": "$name", "drafting": "dspark", "parallel": $streams}
EOF
            STRATA_DEBUG=1 PYTHONPATH=$control:$control/tools "$python" "$server" --engine strata \
                --config "$out/$name.json" --port 8242 --no-api-key >"$out/server-$name-$streams.log" 2>&1 &
            pid=$!
            until python3 -c 'import urllib.request as u; u.urlopen("http://127.0.0.1:8242/v1/models")' 2>/dev/null; do
                kill -0 "$pid" 2>/dev/null || { tail -30 "$out/server-$name-$streams.log"; exit 1; }
                sleep 2
            done
            for cell in same different; do
                python3 tools/bench_concurrent.py http://127.0.0.1:8242 "$name" --levels "$streams" --tokens 128 \
                    --reps 3 --temperatures 0 --prompts "$out/$cell.json" --label "$name $cell" \
                    --output "$out/http-$name-$streams-$cell.json" >"$out/http.log"
                summary "$out/http-$name-$streams-$cell.json" ||
                    { grep '^\[strata\]' "$out/server-$name-$streams.log" | tail -12; exit 1; }
            done
            # the engine's INFO line, which the server keeps only in /metrics; a missing dtype fails the job
            kv=$(python3 "$here/metrics-kv.py" http://127.0.0.1:8242/metrics)
            echo "[kv] $name at $streams streams, HTTP: kv_dtype=$kv"
            stop
        done
    done
}
if [ -n "${STRATA_MEASURE:-}" ]; then
    measure
    exit
fi

started=$(date +%s)
for file in "$engine" "$server" "$python" "$release/tokenizer.json" "$release/inkling_tokenizer.py" "$sq20" "$sq10" \
    "$toolchain"; do
    [ -n "$file" ] && [ -f "$file" ] ||
        { echo "missing staged input: ${file:-engine, server or Control Python in $release}" >&2; exit 1; }
done
echo "[inputs] engine=$engine control=$server sq20=$sq20 sq10=$sq10 tokenizer=$release/tokenizer.json"
# the packs' digests as Strata pins them (terminal/strata_control/inkling_family.py at v0.2.19)
sha256sum -c --quiet <<EOF
0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97  $sq20
ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934  $sq10
7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6  $release/tokenizer.json
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $toolchain
EOF
tar -C /work -xJf "$toolchain"
loader=$(find "/work/$gcc" -name ld-linux-aarch64.so.1 | head -1)
libc=$(dirname "$(find "/work/$gcc" -name libc.so.6 | head -1)")
libstdcxx=$(dirname "$(find "/work/$gcc" -name libstdc++.so.6 | head -1)")
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
[settings] own bench: --bench --native PACK --parallel N, nothing else set.
  KV dtype: --kv auto, fp16 below 16 slots and fp8 from 16 (each [bench] line prints the one used).
  --parallel N: N slots, every slot decodes one token a step in one batched forward.
  prompt: 64 synthetic token ids a slot (--prompt-len's default), id 100 + (31*slot + position) % 1000: the same
  ramp shifted 31 ids a slot, not text and not one identical prompt. output: 32 greedy tokens a slot (--gen default).
  drafting: DSpark on (the default); its tree runs at 1 slot only. graphs: the engine's default, no flag set.
  warmup: none, each run is one fresh process whose only pass is timed. 3 runs a cell, the median counts.
[settings] HTTP bench: Strata's server (strata-control 0.1.65, serve/server.py) with the run config its setup
  writes ("parallel": N slots, "drafting": "dspark", --max-context 32768, its default), under
  tools/bench_concurrent.py: N requests at once on /v1/chat/completions, 128 tokens, temperature 0, ignore_eos,
  --reps 3 (the tool's medians), one 16-token warm-up request a cell. Two cells a level, from the chat prompts of
  tools/bench_prompts.json (its raw-completion prompts need /v1/completions, a route Strata's server does not
  have): "same" sends the first on every stream, "different" sends request i prompt i % their count. Their text
  and names follow; each [http] line gives its cell's prompt tokens, output tokens and completion classes, and each
  [kv] line the KV dtype the engine reported at start (its INFO kv_dtype, from the server's /metrics).
  rate: end to end on the client's clock, every stream's usage tokens over the time from the first request sent to
  the last reply done, prefill included. Strata's server renders Inkling's thinking off (serve/inkling_dialect.py),
  yet the model opens <|content_thinking|> anyway and the server drops that span (InklingDialect._emit_typed), so
  a 128-token reply can stream no text ("no_output") and there are no token arrivals to time decode by. The tokens
  are still generated and counted in usage, so the end-to-end rate stands; a stream decode rate does not.
EOF

python3 - "$out" <<'EOF'
import json, sys
chat = [q for q in json.load(open("tools/bench_prompts.json")) if q["kind"] == "chat"]
for cell, prompts in (("same", chat[:1]), ("different", chat)):
    json.dump(prompts, open(f"{sys.argv[1]}/{cell}.json", "w"))
print("[settings] same prompt, %s: %r" % (chat[0]["name"], chat[0]["prompt"]))
print("[settings] different prompts (%d): %s" % (len(chat), ", ".join("%s=%r" % (q["name"], q["prompt"]) for q in chat)))
EOF
for name in $packs; do
    eval "pack=\$$name"
    # as Control's setup does where its Python reads no zstd: the engine exports the pack's template, and its
    # tokenizer when the pack embeds one; otherwise the pack repo's tokenizer.json is used
    "$strata" --export-tokenizer "$pack" "$out/export-$name"
    tokenizer=$out/export-$name/tokenizer.hf.json
    [ -f "$tokenizer" ] || tokenizer=$release/tokenizer.json
    "$python" "$release/inkling_tokenizer.py" --hf-json "$tokenizer" \
        --template "$out/export-$name/chat_template.jinja" --out "$out/tokenizer-$name"
done
echo "build: $(($(date +%s) - started)) s"
# One measurement a timed phase, so every phase fits the slot's 600 s bound while the job's env still selects the
# cells (the wrappers' STRATA_PACK/STRATA_STREAMS/STRATA_CELL and the halves STRATA_OWN/STRATA_HTTP switch off).
halves=
own && [ "${STRATA_OWN:-1}" = 1 ] && halves=own
http && [ "${STRATA_HTTP:-1}" = 1 ] && halves="$halves http"
status=0
for half in $halves; do
    for name in $packs; do
        for streams in $levels; do
            started=$(date +%s)
            STRATA_MEASURE=1 STRATA_CELL=$half STRATA_PACK=$name STRATA_STREAMS=$streams \
                timeout -k 5 600 sh "$0" || { status=$?; break 2; }
            echo "measure $half $name $streams: $(($(date +%s) - started)) s"
        done
    done
done
exit "$status"
