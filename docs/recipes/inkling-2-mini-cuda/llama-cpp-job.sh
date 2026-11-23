#!/bin/sh
# llama.cpp b12628 serving Inkling 2 Mini (ggml-org IQ4_NL) on TOP's GB10 with the harness and settings of
# TensorFold 188's run-script.sh, so its 1- and 4-stream numbers sit next to ours:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/llama-cpp-job.sh
# The job has no network: the Desk staged llama-cpp-stage.sh's folder (binaries, GGUF, pinned SHA256SUMS) in the pilot
# folder. The release is built on glibc 2.38 and the job image's is older, so the binaries run under the loader and
# glibc of the Arm GNU 14.3 sysroot that ../minimax-h3-cuda/toolchain-stage.sh staged; the host is unchanged. The
# server runs the mainline recipe (all layers on the GPU, 4 slots of 8192 tokens, the GGUF's chat template);
# llama-bench adds its own tg128/pp512 line.
# Logs and receipts land in /work/out/inkling2-llama-cpp.
set -eu
tag=b12628
pilot=/assets/drama3-pilot
stage=$pilot/llama-cpp-$tag
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
gguf=$stage/Inkling-2-Mini-IQ4_NL.gguf
out=/work/out/inkling2-llama-cpp
[ -f "$gguf" ] || { echo "missing $gguf: the Desk stages llama-cpp-stage.sh's folder" >&2; exit 1; }
(cd "$stage" && sha256sum -c --quiet SHA256SUMS)
echo "7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $pilot/cuda-build-toolchain/$gcc.tar.xz" | sha256sum -c --quiet
rm -rf "$out" /work/llama-cpp
mkdir -p "$out" /work/llama-cpp
for archive in "$stage"/*.tar.gz; do tar -C /work/llama-cpp -xzf "$archive"; done
tar -C /work/llama-cpp -xJf "$pilot/cuda-build-toolchain/$gcc.tar.xz"
bin=$(dirname "$(find /work/llama-cpp -name llama-server -type f | head -1)")
ld=$(find /work/llama-cpp/$gcc -name ld-linux-aarch64.so.1 | head -1)
libs=$bin:$(dirname "$(find /work/llama-cpp -name 'libcudart.so*' | head -1)")
for lib in libc.so.6 libstdc++.so.6; do libs=$libs:$(dirname "$(find /work/llama-cpp/$gcc -name $lib | head -1)"); done
libs=$libs:$(dirname "$(ldconfig -p | awk '/libcuda.so.1 /{print $NF; exit}')")
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
# ggml loads its backends from the executable's or the working directory; under an explicit loader only the latter holds
(cd "$bin" && "$ld" --library-path "$libs" ./llama-bench -m "$gguf" -ngl 99 -fa 1 -p 512 -n 128 -r 3) \
    >"$out/llama-bench.txt" 2>&1 || { cat "$out/llama-bench.txt" >&2; exit 1; }
cat "$out/llama-bench.txt"
(cd "$bin" && exec "$ld" --library-path "$libs" ./llama-server -m "$gguf" -ngl 99 -c 32768 -np 4 --jinja \
    --host 127.0.0.1 --port 8189 --alias inkling-2-mini) >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || :' EXIT
python3 - <<'EOF'
import time, urllib.request
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        urllib.request.urlopen("http://127.0.0.1:8189/health", timeout=min(5, max(0.1, deadline - time.monotonic())))
        raise SystemExit(0)
    except OSError:
        time.sleep(min(1, max(0, deadline - time.monotonic())))
raise SystemExit("llama-server did not answer /health in 10 min")
EOF
for level in 1 4; do
    python3 tools/bench_concurrent.py http://127.0.0.1:8189 inkling-2-mini --levels "$level" --tokens 256 --reps 3 \
        --temperatures 0 --label "llama-cpp-gb10-$level" --output "$out/streams-$level.json"
done
python3 - "$out" <<'EOF'
import json, sys
agg = {}
for level in (1, 4):
    cells = json.load(open(f"{sys.argv[1]}/streams-{level}.json"))["cells"]
    if any(c.get("failed") for c in cells):
        raise SystemExit(f"{level} streams: a request failed")
    agg[level] = max(c["aggregate_tps"] for c in cells)
    for c in cells:
        r = c["request"]
        print(f"RESULT llama.cpp {level} stream(s) {c['prompt']}: first token {r['ttft_s']} s, prompt {r['prompt_ttft_tps']} tok/s, decode {c['per_stream_tps']} tok/s a stream, {c['aggregate_tps']} total")
print(f"RESULT llama.cpp aggregate decode: 1 stream {agg[1]:.1f} tok/s, 4 streams {agg[4]:.1f} tok/s ({agg[4] / agg[1]:.2f}x)")
EOF
