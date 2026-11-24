#!/bin/sh
# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, revision ed40e7da62) whole on TOP's GB10:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/run-script.sh
# The job has no network: the Desk put the checkpoint in the pilot folder as Inkling-2-Mini, and the toolchain
# archives of ../minimax-h3-cuda/toolchain-stage.sh beside it. Logs and the 1/2/4/8-stream receipt land in
# /work/out/inkling2. Card 257 pairs our plain-decode build with Mia's DGX Spark recipe (README.md beside this).
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
# TOP's GPU slot is 10 minutes of measurement: the build runs first, then the script again under timeout, with its status
if [ -z "${INKLING2_TIMED:-}" ]; then
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
    INKLING2_TIMED=1 exec timeout 600 sh "$0" "$@"
fi
bin=/work/sm121/bin
rm -rf "$out"
mkdir -p "$out"
"$bin/tf-cuda-test" kernel-contract "inkling2 loop"
# #4387's serving checks take any 4,096+ ids; these are fixed so every run sees the same prompt
python3 -c 'import random; r = random.Random(4387); print(" ".join(str(r.randrange(1000, 199000)) for _ in range(4200)))' >"$out/ids.txt"
# case=1 (sampled) is a known engine defect owned by card 263: its status is printed and returned at the end, so the
# reference, control and serve steps still run and the job stays red
serving=0
"$bin/tf-cuda-test" inkling2-serving "$model" "$out/ids.txt" || serving=$?
echo "RESULT inkling2-serving exit $serving"
text="The three laws of thermodynamics, explained for a curious student:"
# the float64 reference must exit 0
"$bin/tf-cuda-test" inkling2-mini-ref "$model" 32 "$text"
# the planted control must go red with its own differing tokens: a load or CUDA failure also exits non-zero
if INKLING2_REF_PLANT=1 "$bin/tf-cuda-test" inkling2-mini-ref "$model" 32 "$text" 2>"$out/control.log"; then
    echo "planted control stayed green" >&2
    exit 1
fi
cat "$out/control.log" >&2
grep -Eq '\(PLANTED: no messages\): [1-9][0-9]* of 32 greedy tokens differ' "$out/control.log" ||
    { echo "planted control failed before comparing tokens" >&2; exit 1; }
mem() { echo "$1: host $(awk '/^MemTotal:/ {t = $2} /^MemAvailable:/ {a = $2} END {printf "used %d MiB, available %d MiB", (t - a) / 1024, a / 1024}' /proc/meminfo); GPU $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; }
mem "before the server"
/work/sm121/native/bin/tensorfold-native serve "$model" --backend cuda --host 127.0.0.1 --port 8188 \
    --name inkling-2-mini --parallel 8 --context 8192 --no-update-check >"$out/server.log" 2>&1 &
server=$!
# peak GPU memory: the server's own compute allocation, sampled every 500 ms for the whole serve
while :; do nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits; sleep 0.5; done >"$out/pool.txt" 2>/dev/null &
sampler=$!
trap 'kill $server $sampler 2>/dev/null || :' EXIT
python3 - <<'EOF'
import time, urllib.request
for _ in range(180):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)
        raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("server did not answer /v1/models in 3 min")
EOF
mem "server loaded"
# prefill probe: one short and one ~4k-token prompt, read from the server's own done line (prompt=, cached=, ttft=)
python3 - "$out/server.log" <<'EOF'
import json, re, sys, time, urllib.request
word = "The quick brown fox jumps over the lazy dog near the river bank while engineers measure prefill throughput. "
for tag, prompt in (("prefill-short", "Explain one tick of a clock."), ("prefill-long", word * 400)):
    before = open(sys.argv[1]).read().count("] done ")
    body = json.dumps({"model": "inkling-2-mini", "prompt": prompt, "max_tokens": 8, "temperature": 0, "ignore_eos": True})
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/v1/completions", body.encode(),
                                                  {"Content-Type": "application/json"}), timeout=120).read()
    time.sleep(0.2)
    line = [l for l in open(sys.argv[1]) if "] done " in l][before]
    m = re.search(r"prompt=(\d+) cached=(\d+) .*ttft=([\d.]+)", line)
    print(f"RESULT inkling2 mini gb10 {tag}: {m[1]} prompt tokens ({m[2]} cached), prefill "
          f"{(int(m[1]) - int(m[2])) / float(m[3]):,.0f} tok/s, first token {m[3]} s")
EOF
# --alone runs each request solo first and hashes every concurrent reply's tokens against it (output agreement);
# each cell's request metrics carry prompt_ttft_tps (prefill) and ttft_s, its counts the failed and unmeasured.
for level in 1 2 4 8; do
    python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels "$level" --tokens 256 --reps 3 \
        --temperatures 0 --alone --label "gb10-$level" --output "$out/streams-$level.json"
    mem "after $level streams"
done
python3 - "$out" "$model" <<'EOF'
import glob, json, re, struct, sys
out, model = sys.argv[1:]
agg = {}
for level in (1, 2, 4, 8):
    cells = json.load(open(f"{out}/streams-{level}.json"))["cells"]
    if any(c.get("failed") or c.get("unmeasured") for c in cells):
        raise SystemExit(f"{level} streams: a request failed or was unmeasured")
    for c in cells:
        alone = c.get("alone") or {}
        # one equal_alone vote per request per rep: level * reps
        if alone.get("unequal") or alone.get("failed") or alone.get("equal") != level * 3 or alone.get("vacuous"):
            raise SystemExit(f"{level} streams {c['prompt']}: greedy alone agreement failed or was vacuous: {alone}")
    agg[level] = max(c["aggregate_tps"] for c in cells)
    for c in cells:
        r = c["request"]
        print(f"RESULT inkling2 mini gb10 {level} stream(s) {c['prompt']}: first token {r['ttft_s']} s, prompt {r['prompt_ttft_tps']} tok/s, decode {c['per_stream_tps']} tok/s a stream, {c['aggregate_tps']} total, alone-agree {c.get('alone')}")
# weight bytes one decoded token reads: the core layers (loop_config) run num_passes times, each visit reads
# num_experts_per_tok of n_routed_experts experts, the unembedding is read whole and the embedding one row
text = json.load(open(f"{model}/config.json"))
text = text.get("text_config", text)
loop = text["loop_config"]
per_token = 0.0
for path in glob.glob(f"{model}/*.safetensors"):
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    for name, t in header.items():
        if name == "__metadata__" or not name.startswith("model.llm.") or name == "model.llm.embed.weight":
            continue
        size = t["data_offsets"][1] - t["data_offsets"][0]
        layer = re.match(r"model\.llm\.layers\.(\d+)\.", name)
        if layer:
            core = loop["core_start"] <= int(layer[1]) < loop["core_end"]
            size *= loop["num_passes"] if core else 1
            if ".mlp.experts." in name:
                size *= text["num_experts_per_tok"] / text["n_routed_experts"]
        per_token += size
bandwidth = 273e9  # GB10's LPDDR5x peak, NVIDIA's DGX Spark specification
print(f"RESULT inkling2 mini gb10 aggregate decode: " + ", ".join(
    f"{n} stream{'s' if n > 1 else ''} {agg[n]:.1f} tok/s" for n in (1, 2, 4, 8)) + f" (8/1 = {agg[8] / agg[1]:.2f}x)")
print(f"RESULT inkling2 mini weight bytes a token {per_token / 1e9:.2f} GB: one stream's bandwidth limit {bandwidth / per_token:.1f} tok/s at 273 GB/s, measured {100 * agg[1] * per_token / bandwidth:.0f}% of it")
if agg[8] < 1.5 * agg[1]:
    raise SystemExit(f"8 streams gave {agg[8]:.1f} tok/s, not clearly more than 1 stream's {agg[1]:.1f}")
EOF
echo "RESULT inkling2 mini gb10 peak GPU memory sampled every 500 ms: $(sort -n "$out/pool.txt" | tail -n 1) MiB"
echo "RESULT inkling2 mini gb10 server peak RSS: $(awk '/^VmHWM:/ {printf "%d MiB", $2 / 1024}' /proc/$server/status)"
exit "$serving"
