#!/bin/sh
# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, revision ed40e7da62) whole on TOP's GB10:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/run-script.sh
# The job has no network: the Desk put the checkpoint in the pilot folder as Inkling-2-Mini, and the toolchain
# archives of ../minimax-h3-cuda/toolchain-stage.sh beside it. Logs and the 1/4-stream receipt land in /work/out/inkling2.
set -eu
pilot=/assets/drama3-pilot
staged=$pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
out=/work/out/inkling2
model=
for dir in "$pilot/Inkling-2-Mini" "$pilot/models/Inkling-2-Mini"; do
    [ -f "$dir/config.json" ] && model=$dir && break
done
if [ -z "$model" ]; then
    echo "no Inkling-2-Mini/config.json in $pilot or $pilot/models; Inkling configs under /assets:" >&2
    find /assets -maxdepth 5 -name config.json -exec grep -l '"inkling2' {} + >&2 || true
    exit 1
fi
[ -d "$staged" ] || { echo "missing $staged: run ../minimax-h3-cuda/toolchain-stage.sh" >&2; exit 1; }
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
echo "model $model: $(cat "$model"/*.safetensors | sha256sum | cut -d' ' -f1)"
zig build install native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/sm121
bin=/work/sm121/bin
rm -rf "$out"
mkdir -p "$out"
# #4387's serving checks take any 4,096+ ids; these are fixed so every run sees the same prompt
python3 -c 'import random; r = random.Random(4387); print(" ".join(str(r.randrange(1000, 199000)) for _ in range(4200)))' >"$out/ids.txt"
"$bin/tf-cuda-test" inkling2-serving "$model" "$out/ids.txt"
text="The three laws of thermodynamics, explained for a curious student:"
"$bin/tf-cuda-test" inkling2-mini-ref "$model" 32 "$text"
INKLING2_REF_PLANT=1 "$bin/tf-cuda-test" inkling2-mini-ref "$model" 32 "$text"
mem() { echo "$1: $(free -m | awk '/^Mem:/ {print "host used " $3 " MiB, available " $7 " MiB"}'); GPU $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; }
mem "before the server"
/work/sm121/native/bin/tensorfold-native serve "$model" --backend cuda --host 127.0.0.1 --port 8188 \
    --name inkling-2-mini --parallel 4 --context 8192 --no-update-check >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || :' EXIT
python3 - <<'EOF'
import time, urllib.request
for _ in range(900):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)
        raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("server did not answer /v1/models in 15 min")
EOF
mem "server loaded"
for level in 1 4; do
    python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels "$level" --tokens 256 --reps 3 \
        --temperatures 0 --label "gb10-$level" --output "$out/streams-$level.json"
    mem "after $level streams"
done
python3 - "$out" <<'EOF'
import json, sys
for level in (1, 4):
    cells = json.load(open(f"{sys.argv[1]}/streams-{level}.json"))["cells"]
    if any(c.get("failed") for c in cells):
        raise SystemExit(f"{level} streams: a request failed")
EOF
