#!/bin/sh
# llama.cpp CUDA baseline for run-script.sh's receipt: Unsloth's 16.6 GB UD-Q4_K_M GGUF on TOP's GB10.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/llama-cpp-baseline.sh
# The job has no network: the Desk staged Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file (renamed .gguf) in the
# pilot folder's inkling-gguf, and llama.cpp b12611 (690b87d88, ggml-org #32704's merge) with CMake 3.31.6's
# linux-aarch64 archive in inkling-llamacpp.
set -eu
pilot=/assets/drama3-pilot
llama=$pilot/inkling-llamacpp
staged=$pilot/cuda-build-toolchain
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
cmake=cmake-3.31.6-linux-aarch64
gguf=$pilot/inkling-gguf/Inkling-2-Mini-UD-Q4_K_M.gguf
out=/work/out/inkling2-llama
sha256sum -c --quiet <<EOF
78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae  $gguf
b4cc788d63112b2749b40627e719eb5d3b8ed8f00c36d77189f4019cfe64bc9e  $llama/$cmake.tar.gz
EOF
tar -C /work -xJf "$staged/$gcc.tar.xz"
tar -C /work -xzf "$llama/$cmake.tar.gz"
cc=/work/$gcc/bin/aarch64-none-linux-gnu
/work/$cmake/bin/cmake -S "$llama/llama.cpp-b12611" -B /work/llama-build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$cc-gcc" -DCMAKE_CXX_COMPILER="$cc-g++" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER="$cc-g++" -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
/work/$cmake/bin/cmake --build /work/llama-build -j "$(nproc)" --target llama-server >/dev/null
rm -rf "$out"
mkdir -p "$out"
mem() { echo "$1: host $(awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2} END {printf "used %d MiB, available %d MiB", (t - a) / 1024, a / 1024}' /proc/meminfo); GPU $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; }
mem "before the server"
# run-script.sh's settings: 4 streams of 8,192 positions each
/work/llama-build/bin/llama-server -m "$gguf" --jinja -ngl 99 -c 32768 --parallel 4 --host 127.0.0.1 --port 8189 \
    --no-webui >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || :' EXIT
python3 - <<'EOF'
import time, urllib.request
for _ in range(600):
    try:
        urllib.request.urlopen("http://127.0.0.1:8189/health", timeout=5)
        raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("llama-server did not answer /health in 10 min")
EOF
mem "server loaded"
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
print(f"RESULT llama.cpp q4_k_m greedy vs float64 reference: first {same} of {len(w)} tokens equal: {got!r}")
EOF
