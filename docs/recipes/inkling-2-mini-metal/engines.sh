#!/bin/bash
# engines.sh [ENGINE...]: Inkling 2 Mini on this Mac's Metal GPU under TensorFold, llama.cpp and Strata v0.2.19 at 1, 4
# and 64 streams. ENGINE is tensorfold, llama.cpp, strata-sq20 or strata-sq10 (default: all four, in that order). Each
# engine runs alone under `lockf -k /tmp/tensorfold-model.lock`: setup (inputs checked, server up) then measure, each
# logged with its duration; every measured run is one tools/bench_concurrent.py call under `timeout 600`.
# Our bench's settings (docs/receipts/inkling-2-mini-metal-210/bench.sh): 256 tokens, greedy, 3 reps (medians), one
# server with 64 slots for every level. Two cells a level, chat only (Strata's server has no /v1/completions):
# "same" sends bench_concurrent's built-in chat prompt on every stream, "diverse" sends request i the i-th chat prompt
# of tools/bench_prompts.json. Inputs come from the Desk's STAGE folder (README.md lists them); nothing is fetched.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
wt=$(git -C "$here" rev-parse --show-toplevel)
model=${MODEL:-$HOME/models/Inkling-2-Mini}
stage=${STAGE:-/Users/agents/projects/capri/desk/jobs/tf250-engines}
out=${OUT:-/tmp/ink2-engines}
python=/opt/homebrew/bin/python3.13
port=18250
command -v timeout >/dev/null || timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; } # macOS ships no timeout

fail() { echo "$*" >&2; exit 1; }
check() { # check SHA256 FILE
    [ -f "$2" ] || fail "missing staged input: $2"
    echo "$1  $2" | shasum -a 256 -c --quiet || fail "digest differs: $2"
}

serve_tensorfold() {
    (cd "$wt" && zig build native -Doptimize=ReleaseFast)
    echo "[settings] tensorfold $(git -C "$wt" rev-parse HEAD): $model, quant $("$python" -c 'import json, sys
print(json.load(open(sys.argv[1]))["quant_method"])' "$model/hf_quant_config.json"), KV bf16 (Metal's only)"
    echo "[settings] flags: serve --context 65536 --parallel 64 --drafter $model (DSpark on)"
    "$wt/zig-out/native/bin/tensorfold-native" serve "$model" --port "$port" --context 65536 --parallel 64 \
        --drafter "$model" >"$log/server.log" 2>&1 &
}

serve_llama() {
    local llama gguf=$stage/gguf/Inkling-2-Mini-IQ4_NL.gguf build
    llama=$(find "$stage/llama.cpp" -name llama-server -type f 2>/dev/null | head -1)
    [ -n "$llama" ] || fail "missing staged input: $stage/llama.cpp/**/llama-server"
    check cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4 "$gguf"
    build=$("$llama" --version 2>&1 | sed -nE 's/^version: ([0-9]+) .*/\1/p; s/^version: .*build ([0-9]+).*/\1/p')
    [ "${build:-0}" -ge 12406 ] || fail "llama.cpp build ${build:-unknown} predates b12406 (Inkling 2): $llama"
    echo "[settings] llama.cpp b$build: $gguf, $(stat -f %z "$gguf") bytes, sha256 cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4"
    echo "[settings] flags: -ngl 99 -fa on -ctk f16 -ctv f16 -c 262144 -np 64 --jinja (4,096 tokens a slot, no drafter)"
    "$llama" -m "$gguf" -ngl 99 -fa on -ctk f16 -ctv f16 -c 262144 -np 64 --jinja --alias Inkling-2-Mini \
        --host 127.0.0.1 --port "$port" >"$log/server.log" 2>&1 &
}

serve_strata() { # serve_strata PACK-NAME: Strata's own server over its engine, as its setup writes the run config
    local strata=$stage/strata/strata-inkling control=$stage/strata/control pack=$stage/strata/Inkling-2-Mini-$1.strata
    local tokenizer=$log/export/tokenizer.hf.json
    check 9bd678185e41526eb3796220c3dc7723bb2d13b60e585241ee4f05e7c492f27b "$strata"
    case $1 in # the packs' digests as Strata v0.2.19 pins them
        SQ-20) check 0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97 "$pack" ;;
        SQ-10) check ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934 "$pack" ;;
    esac
    [ -f "$control/serve/server.py" ] || fail "missing staged input: $control (strata-control-0.1.65-macos-arm64)"
    [ "$("$strata" --version)" = "strata-inkling 0.1.65" ] || fail "not Strata v0.2.19's engine: $strata"
    "$strata" --export-tokenizer "$pack" "$log/export"
    [ -f "$tokenizer" ] || tokenizer=$stage/strata/tokenizer.json
    "$control/python/bin/python3" "$control/tools/inkling_tokenizer.py" --hf-json "$tokenizer" \
        --template "$log/export/chat_template.jinja" --out "$log/tokenizer"
    cat >"$log/config.json" <<EOF
{"exe": "$strata", "args": ["--native", "$pack", "--max-context", "32768"], "cwd": "$log", "arch": "inkling2_loop",
 "tokenizer": "$log/tokenizer", "model_name": "Inkling-2-Mini", "drafting": "dspark", "parallel": 64}
EOF
    echo "[settings] strata-inkling 0.1.65 (Strata v0.2.19, macos-arm64-metal): $pack"
    echo "[settings] flags: $(tr -d '\n' <"$log/config.json"); KV dtype --kv auto (the server log's kv_dtype follows)"
    STRATA_DEBUG=1 PYTHONPATH=$control:$control/tools "$control/python/bin/python3" "$control/serve/server.py" \
        --engine strata --config "$log/config.json" --port "$port" --no-api-key >"$log/server.log" 2>&1 &
}

run() { # run ENGINE: setup, then 1, 4 and 64 streams on both cells
    local engine=$1 started pid streams cell
    log=$out/$engine
    rm -rf "$log" && mkdir -p "$log"
    ! curl -s "http://127.0.0.1:$port/v1/models" >/dev/null || fail "port $port already serves; stop that server first"
    started=$(date +%s)
    case $engine in
        tensorfold) serve_tensorfold ;;
        llama.cpp) serve_llama ;;
        strata-sq20) serve_strata SQ-20 ;;
        strata-sq10) serve_strata SQ-10 ;;
        *) fail "unknown engine: $engine" ;;
    esac
    pid=$!
    trap "kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null" EXIT
    for _ in $(seq 600); do
        curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && break
        kill -0 "$pid" 2>/dev/null || { tail -30 "$log/server.log"; fail "$engine server exited during setup"; }
        sleep 1
    done
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null || fail "$engine server not ready in 600 s"
    echo "[$engine] setup: $(($(date +%s) - started)) s"
    started=$(date +%s)
    for streams in 1 4 64; do
        for cell in same diverse; do
            timeout 600 "$python" "$wt/tools/bench_concurrent.py" "http://127.0.0.1:$port" Inkling-2-Mini \
                --levels "$streams" --tokens 256 --reps 3 --temperatures 0 --prompts "$out/prompts/$cell.json" \
                --label "$engine $cell" --output "$log/$streams-$cell.json" >"$log/$streams-$cell.log" 2>&1 ||
                { tail -20 "$log/$streams-$cell.log"; fail "$engine $streams streams $cell: bench failed or passed 600 s"; }
        done
    done
    grep -ho 'kv_dtype=[a-z0-9]*' "$log/server.log" | sort -u || :
    echo "[$engine] measure: $(($(date +%s) - started)) s"
}

if [ "${1-}" = run ]; then
    run "$2"
    exit
fi
[ -e /Users/Shared/agent-work/mac-gpu-window ] || { echo "no Mac GPU window" >&2; exit 3; }
engines=("$@")
[ $# -gt 0 ] || engines=(tensorfold llama.cpp strata-sq20 strata-sq10)
mkdir -p "$out/prompts"
PYTHONPATH=$wt/tools "$python" - "$wt" "$out/prompts" <<'EOF'
import json, sys
from bench_concurrent import PROMPTS
chat = [q for q in json.load(open(f"{sys.argv[1]}/tools/bench_prompts.json")) if q["kind"] == "chat"]
same = [q for q in PROMPTS if q["kind"] == "chat"]
for cell, prompts in (("same", same), ("diverse", chat)):
    json.dump(prompts, open(f"{sys.argv[2]}/{cell}.json", "w"))
    print(f"[settings] {cell}: " + "; ".join(f"{q['name']}={q['prompt']!r}" for q in prompts))
EOF
status=0 completed=()
for engine in "${engines[@]}"; do
    if lockf -k /tmp/tensorfold-model.lock "$0" run "$engine" 2>&1 | tee "$out/$engine.log"; then
        completed+=("$engine")
    else
        status=1
    fi
done
[ ${#completed[@]} -gt 0 ] || exit "$status"
"$python" - "$out" "${completed[@]}" <<'EOF' # only engines this run completed, so stale results in OUT cannot vote
import glob, json, os, sys
fmt = lambda v: "-" if v is None else f"{v:.1f}"
best = {}
for path in sorted(p for e in sys.argv[2:] for p in glob.glob(f"{sys.argv[1]}/{e}/[0-9]*-*.json")):
    engine, cell = os.path.basename(os.path.dirname(path)), path.rsplit("-", 1)[1][:-5]
    for c in json.load(open(path))["cells"]:
        print(f"RESULT {engine} {c['streams']} streams {cell}: {fmt(c['aggregate_tps'])} tok/s total, "
              f"{fmt(c['per_stream_tps'])} a stream (medians of 3), failed {c['failed']}, unmeasured {c['unmeasured']}")
        key = (c["streams"], cell)
        if not c["failed"] and not c["unmeasured"] and (c["aggregate_tps"] or 0) > best.get(key, ("", 0))[1]:
            best[key] = (engine, c["aggregate_tps"])
for (streams, cell), (engine, tps) in sorted(best.items()):
    print(f"FASTEST {streams} streams {cell}: {engine}, {tps:.1f} tok/s total")
EOF
exit "$status"
