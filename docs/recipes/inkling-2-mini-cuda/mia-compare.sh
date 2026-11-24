#!/bin/sh
# Card 256: Mia's published recipe against our Inkling build, paired on TOP's GB10.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/mia-compare.sh
#
# Both arms serve the same checkpoint with the same client, prompts and greedy settings, serially. Mia's arm is
# TensorFold dc2ca8a1 with patches 0001-0017 from MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold@6061fa89, built with
# the Dockerfile's flags (no network here, so no image: the tree is staged). Our arm is this checkout (c99eb630d),
# built with docs/recipes/inkling-2-mini-cuda/setup.sh's flags.
#
# The job has no network: the Desk stages, under /assets/drama3-pilot,
#   mia-inkling/      Mia's repository at 6061fa89 (patches/, scripts/config.sh, README.md; .git not needed)
#   tf-dc2ca8a1/      TensorFold's tree at dc2ca8a1 (no .git)
#   Inkling-2-Mini/   the checkpoint at upstream revision 7a67ae23 (Mia's pin: shard 65e60f51..., MTP 7c3b6513...)
#   cuda-build-toolchain/  zig 0.17.0 and Arm GNU 14.3 archives (../minimax-h3-cuda/toolchain-stage.sh)
# The checkpoint gate below stops the job when the staged folder is another revision: both arms must read the same bytes.
#
# Matched settings: --context 131072 --parallel 8, bf16 KV, drafts off, greedy, thinking on. Mia's published table
# used 262144/fp8/DSpark/262k, which this head of ours cannot run (no fp8 KV for Mini CUDA; --drafter refused), so
# the paired run holds the common configuration and her own levers are reported from her README, not measured here.
set -eu
. docs/recipes/inkling-2-mini-cuda/toolchain.sh
out=/work/out/mia-compare
rm -rf "$out"
mkdir -p "$out"
mia=$pilot/mia-inkling
tree=$pilot/tf-dc2ca8a1
client=docs/recipes/inkling-2-mini-cuda/mia-compare.py

fail() { echo "STOP: $*" >&2; exit 1; }

# Mia's recipe, pinned by the bytes the job reads: README, settings, and the 17 patches in order.
[ "$(sha256sum "$mia/README.md" | cut -d' ' -f1)" = 2bcb66e232f2038efffd5dfd3baa1e17e278471df73dd9aafe0097853943eefe ] ||
    fail "$mia/README.md is not Mia's at 6061fa89"
[ "$(sha256sum "$mia/scripts/config.sh" | cut -d' ' -f1)" = 04a1fe38db7f45bcb7884515628fe6e4d9a2001f538ab1686f58e097192938b9 ] ||
    fail "$mia/scripts/config.sh is not Mia's at 6061fa89"
[ "$(cat "$mia"/patches/*.patch | sha256sum | cut -d' ' -f1)" = ab0e476de064dc42dd7ebd25534f2059c54cd3430c4e867db08a179f11723b2d ] ||
    fail "$mia/patches is not patches 0001-0017 of 6061fa89"
[ -d "$tree" ] || fail "missing $tree: the Desk must stage TensorFold's dc2ca8a1 tree"

shard=$(sha256sum "$model/model-00001-of-00001.safetensors" | cut -d' ' -f1)
mtp=$(sha256sum "$model/mtp.safetensors" | cut -d' ' -f1)
[ "$shard" = 65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1 ] ||
    fail "$model's shard is $shard, not Mia's pinned 65e60f51 (revision 7a67ae23): stage that revision"
[ "$mtp" = 7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2 ] ||
    fail "$model's mtp.safetensors is $mtp, not 7c3b6513"
echo "checkpoint $model: shard $shard, MTP $mtp"

# Mia's arm: her tree, her 17 patches in order, the Dockerfile's build flags.
rm -rf /work/mia-c4
mkdir -p /work/mia-c4
cp -a "$tree/." /work/mia-c4/
git -C /work/mia-c4 init -q
for p in "$mia"/patches/*.patch; do
    git -C /work/mia-c4 apply --check "$p"
    git -C /work/mia-c4 apply "$p"
done
( cd /work/mia-c4 && zig build fatbins native -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast \
    -Dnative-fp-decode=true --prefix /work/mia )
# Our arm: this checkout (c99eb630d), our recipe's build flags.
zig build install native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/ours
echo "built Mia's dc2ca8a1+patches and ours at $(git rev-parse HEAD 2>/dev/null || echo unknown)"

mem() {
    echo "MEM $1: MemAvailable $(awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo) GiB;" \
        "compute apps $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';' || echo none)"
}

# Serve one arm, run the three phases under the slot's bound, and report the server's peak RSS and GPU memory.
arm() {
    label=$1 bin=$2
    shift 2
    log=$out/$label
    mem "before $label"
    "$bin" "$@" >"$log.server.log" 2>&1 &
    pid=$!
    trap 'kill $pid 2>/dev/null || :' EXIT
    python3 - <<'EOF'
import time, urllib.request
for _ in range(900):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)
        raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit("the server did not answer /v1/models in 15 min")
EOF
    mem "$label loaded"
    for phase in decode prefill agree; do
        case $phase in
            decode) bound=${DECODE_S:-900} ;;
            prefill) bound=${PREFILL_S:-1800} ;;
            agree) bound=${AGREE_S:-600} ;;
        esac
        timeout "$bound" python3 "$client" --url http://127.0.0.1:8188 --out "$log.jsonl" \
            --phase "$phase" || echo "PHASE $phase of $label exited $?"
        mem "$label after $phase"
    done
    awk '/^VmHWM:/ {print "PEAK '"$label"' server RSS " $2 " kB"}' "/proc/$pid/status"
    kill "$pid" 2>/dev/null || :
    wait "$pid" 2>/dev/null || :
    trap - EXIT
    sleep 5
}

arm mia /work/mia/native/bin/tensorfold-native serve \
    --name inkling-2-mini --host 127.0.0.1 --port 8188 \
    --context 131072 --parallel 8 --max-tokens 32768 --no-drafts
arm ours /work/ours/native/bin/tensorfold-native serve \
    --backend cuda --name inkling-2-mini --host 127.0.0.1 --port 8188 \
    --context 131072 --parallel 8 --no-update-check

python3 - "$out" <<'EOF'
import json, statistics, sys
out = sys.argv[1]
arms = {name: [json.loads(line) for line in open(f"{out}/{name}.jsonl")] for name in ("mia", "ours")}
bad = 0

def rows(name, kind):
    return [r for r in arms[name] if r["kind"] == kind]

for name in arms:
    failed = sum(r["failed"] for r in arms[name])
    print(f"RESULT {name}: {len(arms[name])} requests, {failed} failed")
    bad += failed

for prompt in ("prose", "code"):
    for level in (1, 2, 4, 8):
        cells = {}
        for name in arms:
            rs = [r for r in rows(name, "decode") if r["prompt"] == prompt and r["streams"] == level
                  and r["aggregate_tps"] is not None]
            cells[name] = rs and (statistics.median(r["aggregate_tps"] for r in rs),
                                  statistics.median(r["per_stream_tps"] for r in rs),
                                  statistics.median(r["ttft_s"] for r in rs), len(rs))
        if all(cells.values()):
            print(f"RESULT decode {prompt} x{level}: mia {cells['mia'][0]:.1f} tok/s total, {cells['mia'][1]:.1f} a"
                  f" stream, TTFT {cells['mia'][2] * 1000:.0f} ms (n={cells['mia'][3]}) | ours"
                  f" {cells['ours'][0]:.1f}, {cells['ours'][1]:.1f}, TTFT {cells['ours'][2] * 1000:.0f} ms"
                  f" (n={cells['ours'][3]})")
        else:
            print(f"RESULT decode {prompt} x{level}: missing ({ {k: bool(v) for k, v in cells.items()} })")
            bad += 1

for want in (4096, 8192, 16384, 32768, 65536, 131000):
    cells = {}
    for name in arms:
        rs = [r for r in rows(name, "prefill") if r["want_tokens"] == want and r["prefill_s"]]
        if rs:
            best = min(rs, key=lambda r: r["prefill_s"])
            fresh = best["prompt_tokens"] - (best["cached_tokens"] or 0)
            cells[name] = (fresh / best["prefill_s"], best["prompt_tokens"], best["ttft_s"], len(rs))
    if all(cells.values()):
        print(f"RESULT prefill ~{want}: mia {cells['mia'][0]:.0f} tok/s on {cells['mia'][1]} tok,"
              f" TTFT {cells['mia'][2]:.2f} s (n={cells['mia'][3]}) | ours {cells['ours'][0]:.0f} tok/s on"
              f" {cells['ours'][1]} tok, TTFT {cells['ours'][2]:.2f} s (n={cells['ours'][3]})")
    else:
        print(f"RESULT prefill ~{want}: missing ({ {k: bool(v) for k, v in cells.items()} })")
        bad += 1

for idx in range(6):
    got = {}
    for name in arms:
        r = next((r for r in rows(name, "agree") if r["prompt_idx"] == idx), None)
        got[name] = None if r is None or r["failed"] else (r["token_sha"], r["text_sha256"])
    if got["mia"] and got["ours"]:
        same = got["mia"][0] is not None and got["mia"][0] == got["ours"][0] or got["mia"][1] == got["ours"][1]
        print(f"RESULT agree {idx}: {'same' if same else 'DIFFERENT'} mia {got['mia']} ours {got['ours']}")
        bad += 0 if same else 1
    else:
        print(f"RESULT agree {idx}: missing ({ {k: bool(v) for k, v in got.items()} })")
        bad += 1

sys.exit(1 if bad else 0)
EOF
