#!/bin/sh
# Card 259's same-job A/B on TOP's GB10: Inkling 2 Mini decode at 1 and 4 streams, main against the card's three
# levers (fused kernels, the gate, CUDA graphs), on the toolchain and checkpoint of 188's run-script.sh:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/bench-259.sh
# The job has no network and may have no git history, so every side is built from this commit. Each lever is one
# commit, kept here as its diff (`git diff C^ C`): bench-259-fusion.patch, bench-259-gate.patch and
# bench-259-graphs.patch, in the order they were committed. The job reverse-applies them newest first to reach main,
# then puts them back one at a time and builds a server after each step into /work/259/SIDE: main, fusion, gate
# (fusion + gate) and all (this commit again). The builds are not capped.
# Each measurement runs as its own job under `timeout 600` and logs "measure NAME: N s": first the no-draft bandwidth
# ceiling from the checkpoint's safetensors headers at 273 GB/s, then one fresh server an arm: control-1 (main), all,
# control-2 (main), gate, fusion, control-3 (main). An arm runs bench_concurrent.py's code and chat prompts, greedy,
# 64 tokens, 3 reps, at 1 and 4 streams; a failed or unmeasured request fails the job. Its figure at a level is the
# median aggregate tok/s over both prompts' reps (188's receipt took 256 tokens and the better prompt, so its 32.5 and
# 117.8 tok/s are not these controls). The card's gain is `all` over its two neighbouring controls. A lever's gain is
# each side over the one below it (fusion over control-3, gate over fusion, all over gate), printed next to the three
# controls' spread, the noise it has to clear. Logs, reports and ledger.jsonl land in /work/out/inkling2-259.
set -eu
here=docs/recipes/inkling-2-mini-cuda
pilot=/assets/drama3-pilot
out=/work/out/inkling2-259
sides=/work/259
tokens=64
reps=3

if [ "${1:-}" = ceiling ]; then
    # weight bytes a round of N streams reads: every non-expert weight once (the core layers once a pass, the
    # unembedding whole, the embedding only a row each, left out) and the experts its N rows route to, a share
    # 1 - (1 - k/E)^N under uniform routing. One stream's figure is a hard limit; four streams' is an estimate,
    # since real routing overlaps more than uniform.
    exec python3 - "$2" "$out/ceiling.json" <<'EOF'
import glob, json, re, struct, sys
model, path = sys.argv[1:]
text = json.load(open(f"{model}/config.json"))
text = text.get("text_config", text)
loop = text["loop_config"]
fixed = experts = 0
for shard in glob.glob(f"{model}/*.safetensors"):
    with open(shard, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    for name, t in header.items():
        if name == "__metadata__" or not name.startswith("model.llm.") or name == "model.llm.embed.weight":
            continue
        size = t["data_offsets"][1] - t["data_offsets"][0]
        layer = re.match(r"model\.llm\.layers\.(\d+)\.", name)
        if layer and loop["core_start"] <= int(layer[1]) < loop["core_end"]:
            size *= loop["num_passes"]
        if layer and ".mlp.experts." in name:
            experts += size
        else:
            fixed += size
if not fixed or not experts:
    raise SystemExit(f"no model.llm weights or no experts in {model}'s safetensors headers")
bandwidth = 273e9  # GB10's LPDDR5x peak, NVIDIA's DGX Spark specification
limits = {}
for n in (1, 4):
    read = fixed + experts * (1 - (1 - text["num_experts_per_tok"] / text["n_routed_experts"]) ** n)
    limits[str(n)] = n * bandwidth / read
    print(f"RESULT inkling2 mini 259 no-draft limit {n} stream(s): {read / 1e9:.2f} GB a round, {limits[str(n)]:.1f} tok/s total at 273 GB/s")
json.dump(limits, open(path, "w"))
EOF
fi

if [ "${1:-}" = arm ]; then
    name=$2
    side=$3
    model=$4
    server=
    stop_server() { # the server drains on SIGTERM; it is never force-killed mid-batch
        [ -n "$server" ] || return 0
        kill "$server" 2>/dev/null || :
        wait "$server" 2>/dev/null || :
        server=
    }
    died() {
        tail -n 20 "$out/server-$name.log" >&2
        exit 1
    }
    trap 'stop_server' EXIT
    trap 'stop_server; exit 130' INT
    trap 'stop_server; exit 143' TERM
    "$sides/$side/native/bin/tensorfold-native" serve "$model" --backend cuda --host 127.0.0.1 --port 8188 \
        --name inkling-2-mini --parallel 4 --context 8192 --no-update-check >"$out/server-$name.log" 2>&1 &
    server=$!
    python3 - "$server" <<'EOF' || died
import sys, time, urllib.request
while True:  # the job's 600 s cap ends a server that never answers
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)
        raise SystemExit(0)
    except OSError:
        # the server is the shell's child, not ours: a zombie or a gone pid has exited
        try:
            state = open(f"/proc/{sys.argv[1]}/stat").read().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            state = "Z"
        if state == "Z":
            raise SystemExit("the server exited before answering /v1/models")
        time.sleep(1)
EOF
    python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels 1,4 --tokens "$tokens" \
        --reps "$reps" --temperatures 0 --label "259-$name" --output "$out/$name.json" >/dev/null || died
    stop_server
    exec python3 - "$out" "$name" "$side" "$reps" <<'EOF'
import json, math, statistics, sys
out, name, side = sys.argv[1:4]
reps = int(sys.argv[4])
cells = json.load(open(f"{out}/{name}.json"))["cells"]
row = {"arm": name, "side": side}
replies = set()
for level in (1, 4):
    got = [c for c in cells if c["streams"] == level]
    if sorted(c["prompt"] for c in got) != ["chat", "code"]:
        raise SystemExit(f"{name}: {level} stream(s) ran {[c['prompt'] for c in got]}, not the code and chat cells")
    for c in got:
        if c["failed"] or c["unmeasured"] or len(c["reps"]) != reps or any(
               r["failed"] or r["unmeasured"] or len(r["per_stream_tps"]) != level or
               not math.isfinite(r["aggregate_tps"]) or r["aggregate_tps"] <= 0 for r in c["reps"]):
            raise SystemExit(f"{name}: {level} stream(s) {c['prompt']}: {c['failed']} failed and {c['unmeasured']} "
                             f"unmeasured requests in {len(c['reps'])} reps, errors {c.get('errors')}")
        replies |= {(c["prompt"], str(e["token_sha256"] or e["output_sha256"])) for r in c["reps"] for e in r["evidence"]}
        q = c["request"]
        print(f"RESULT inkling2 mini 259 {name} ({side}) {level} stream(s) {c['prompt']}: first token {q['ttft_s']} s, "
              f"decode {c['per_stream_tps']} tok/s a stream, {c['aggregate_tps']} total")
    row[str(level)] = round(statistics.median(r["aggregate_tps"] for c in got for r in c["reps"]), 1)
row["replies"] = sorted(replies)
with open(f"{out}/ledger.jsonl", "a") as ledger:
    ledger.write(json.dumps(row) + "\n")
EOF
fi

staged=$pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
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
rm -rf "$out" "$sides"
mkdir -p "$out" "$sides"
job() { # job LABEL MODE ARGS...: this script's MODE as one measurement under its own 10-minute cap
    label=$1
    shift
    start=$(date +%s)
    status=0
    timeout --kill-after=20 600 sh "$0" "$@" || status=$?
    echo "measure $label: $(($(date +%s) - start)) s"
    [ "$status" != 124 ] && [ "$status" != 137 ] || echo "measure $label: cut off at 600 s" >&2
    return "$status"
}
job ceiling ceiling "$model"

start=$(date +%s)
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
# caches and installs go to /work; the checkout itself only takes the patches
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
# reading every shard also fills the page cache, so the first arm's server loads as warm as the later ones
echo "model $model: $(cat "$model"/*.safetensors | sha256sum | cut -d' ' -f1)"
# a lever's commit may also touch this folder, and sh reads this script as it runs: the patches skip it
lever() { git -c safe.directory='*' apply "$@" --exclude="$here/*" "$here/bench-259-$lever_name.patch"; }
# the levers out of the tree, oldest first
reverted=
restore() { for lever_name in $reverted; do lever || :; done; }
trap 'restore' EXIT
for lever_name in graphs gate fusion; do # every patch has to reverse before the first build is paid for
    lever -R
    reverted="$lever_name $reverted"
done
for side in main fusion gate all; do
    if [ "$side" != main ]; then
        lever_name=${reverted%% *}
        lever
        reverted=${reverted#* }
    fi
    zig build native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix "$sides/$side"
done
echo "build: $(($(date +%s) - start)) s"

for arm in control-1:main all:all control-2:main gate:gate fusion:fusion control-3:main; do
    job "${arm%%:*}" arm "${arm%%:*}" "${arm##*:}" "$model"
done
python3 - "$out" <<'EOF'
import json, sys
out = sys.argv[1]
rows = {r["arm"]: r for r in map(json.loads, open(f"{out}/ledger.jsonl"))}
limits = json.load(open(f"{out}/ceiling.json"))
levels = ("1", "4")
for name in ("control-1", "all", "control-2", "gate", "fusion", "control-3"):
    r = rows[name]
    same = "the same replies as" if r["replies"] == rows["control-1"]["replies"] else "REPLIES DIFFER from"
    print(f"RESULT inkling2 mini 259 {name} ({r['side']}): 1 stream {r['1']} tok/s ({r['1'] / limits['1']:.0%} of the "
          f"limit), 4 streams {r['4']} tok/s ({r['4'] / limits['4']:.0%} of the estimate), {same} control-1")
spread = {}
for level in levels:
    got = [rows[f"control-{i}"][level] for i in (1, 2, 3)]
    spread[level] = max(got) / min(got) - 1
    print(f"RESULT inkling2 mini 259 controls at {level} stream(s): {' / '.join(map(str, got))} tok/s, spread {spread[level]:.1%}")
# the skill's rule: over 3% above both neighbouring controls at a level, and nowhere over 3% below either
card, before, after = rows["all"], rows["control-1"], rows["control-2"]
gain = {k: card[k] / max(before[k], after[k]) - 1 for k in levels}
worst = {k: card[k] / min(before[k], after[k]) - 1 for k in levels}
kept = any(g > 0.03 for g in gain.values()) and all(w > -0.03 for w in worst.values())
print("RESULT inkling2 mini 259 card (all between control-1 and control-2): " + ", ".join(
      f"{k} stream(s) {card[k]} vs {before[k]}/{after[k]} tok/s ({gain[k]:+.1%} over the better control)" for k in levels) +
      f": {'clears both controls by over 3%' if kept else 'does not clear both controls by 3%'}")
for lever, above, below in (("fusion", "fusion", "control-3"), ("gate", "gate", "fusion"), ("graphs", "all", "gate")):
    print(f"RESULT inkling2 mini 259 lever {lever} ({above} over {below}): " + ", ".join(
          f"{k} stream(s) {rows[above][k]} vs {rows[below][k]} tok/s ({rows[above][k] / rows[below][k] - 1:+.1%}, "
          f"controls' spread {spread[k]:.1%})" for k in levels))
EOF
