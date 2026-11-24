#!/bin/sh
# llama.cpp CUDA baseline for run-script.sh's receipt: Unsloth's 16.6 GB UD-Q4_K_M GGUF on TOP's GB10, served by the
# llama-server llama-cpp-build.sh left in /work/llama-build, on the staged toolchain's loader. compare.sh runs both.
set -eu
pilot=/assets/drama3-pilot
gguf=$pilot/inkling-gguf/Inkling-2-Mini-UD-Q4_K_M.gguf
out=/work/out/inkling2-llama
rm -rf "$out"
mkdir -p "$out"
mem() { echo "$1: host $(awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2} END {printf "used %d MiB, available %d MiB", (t - a) / 1024, a / 1024}' /proc/meminfo); GPU $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; }
mem "before the server"
# run-script.sh's settings: 4 streams of 8,192 positions each
sh /work/llama-build/llama-server.sh -m "$gguf" --jinja -ngl 99 -c 32768 --parallel 4 --host 127.0.0.1 --port 8189 \
    --no-webui >"$out/server.log" 2>&1 &
server=$!
trap 'status=$?; kill $server 2>/dev/null || :; [ $status = 0 ] || { echo "llama-server log, last 60 lines:"; tail -n 60 "$out/server.log"; }' EXIT
until python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8189/health", timeout=5)' 2>/dev/null; do
    if ! kill -0 "$server" 2>/dev/null; then
        wait "$server" && code=0 || code=$?
        echo "llama-server exited with status $code before answering /health" >&2
        exit 1
    fi
    sleep 1
done
mem "server loaded"
# the receipt names what the server did with -c and -ngl
grep -m 2 -E 'n_ctx_slot|offloaded' "$out/server.log" || :
# The bench warm-up fills llama.cpp's slot cache; measure the code prompt cold before it.
python3 - <<'EOF'
import json, statistics, sys, time, urllib.request
sys.path.insert(0, "tools")
from bench_concurrent import PROMPTS
prompt = next(p["prompt"] for p in PROMPTS if p["name"] == "code")
rates = []
for _ in range(3):
    body = json.dumps({"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": False}).encode()
    request = urllib.request.Request("http://127.0.0.1:8189/completion", body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        tokens = json.load(response)["timings"]["prompt_n"]
    rates.append(tokens / (time.perf_counter() - start))
print(f"RESULT llama.cpp q4_k_m gb10 1 stream code prompt, uncached: {tokens} tokens, median {statistics.median(rates):.1f} tok/s (prompt tokens over the one-token reply's time)")
EOF
for level in 1 4; do
    python3 tools/bench_concurrent.py http://127.0.0.1:8189 inkling-2-mini --levels "$level" --tokens 256 --reps 3 \
        --temperatures 0 --label "llama-$level" --output "$out/streams-$level.json"
    mem "after $level streams"
done
python3 - "$out" <<'EOF'
import json, sys, urllib.request
out = sys.argv[1]
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:8189{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))
agg = {}
for level in (1, 4):
    cells = json.load(open(f"{out}/streams-{level}.json"))["cells"]
    if any(c.get("failed") for c in cells):
        raise SystemExit(f"{level} streams: a request failed")
    agg[level] = max(c["aggregate_tps"] for c in cells)
    for c in cells:
        r = c["request"]
        print(f"RESULT llama.cpp q4_k_m gb10 {level} stream(s) {c['prompt']}: first token {r['ttft_s']} s, prompt {r['prompt_ttft_tps']} tok/s, decode {c['per_stream_tps']} tok/s a stream, {c['aggregate_tps']} total")
print(f"RESULT llama.cpp q4_k_m gb10 aggregate decode: 1 stream {agg[1]:.1f} tok/s, 4 streams {agg[4]:.1f} tok/s ({agg[4] / agg[1]:.2f}x)")
# run-script.sh's float64 reference continuation of this prompt (job top-16bffb6ca094: TensorFold equals it on all 32)
prompt = "The three laws of thermodynamics, explained for a curious student:"
want = "\n\n  1. Energy cannot be created or destroyed, only transformed.\n  2. Entropy (disorder) in a closed system always increases."
got = post("/completion", {"prompt": prompt, "n_predict": 32, "temperature": 0, "cache_prompt": False})["content"]
ids = lambda s: post("/tokenize", {"content": s})["tokens"]
w, g = ids(want), ids(got)
same = next((i for i, (a, b) in enumerate(zip(w, g)) if a != b), min(len(w), len(g)))
print(f"RESULT llama.cpp q4_k_m greedy vs float64 reference, as text (re-tokenized): first {same} of {len(w)} tokens equal: {got!r}")
# teacher-forced, as tf-cuda-test inkling2-mini-ref scores TensorFold: after the reference's own prefix, is llama.cpp's
# next pick the reference's token? A free run says nothing past its first difference.
start = post("/tokenize", {"content": prompt, "add_special": True})["tokens"]
piece = lambda t: post("/detokenize", {"tokens": [t]})["content"]
pick = lambda i: post("/completion", {"prompt": start + w[:i], "n_predict": 1, "temperature": 0, "cache_prompt": False})["content"]
misses = [i for i in range(len(w)) if pick(i) != piece(w[i])]
print(f"RESULT llama.cpp q4_k_m teacher-forced on the float64 reference: {len(w) - len(misses)} of {len(w)} picks equal, differing at {misses}")
EOF
