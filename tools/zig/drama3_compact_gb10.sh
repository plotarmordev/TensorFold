#!/bin/sh
# Drama 3 compact NVFP4 on one GB10, the checks host tests cannot make; RECEIPT.txt and every log go to RECEIPT_DIR.
set -eu
[ "$#" -eq 5 ] || { echo 'usage: drama3_compact_gb10.sh FP8_MODEL COMPACT_MODEL CODEC NVCC RECEIPT_DIR' >&2; exit 2; }
fp8=$1 compact=$2 codec=$3 nvcc=$4 out=$5
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
base=$(git -C "$root" merge-base HEAD origin/main)
main_tree=$(mktemp -d "${TMPDIR:-/tmp}/drama3-main.XXXXXX")
trap 'git -C "$root" worktree remove --force "$main_tree"' EXIT
git -C "$root" worktree add --detach "$main_tree" "$base" >/dev/null
for tree in "$root" "$main_tree"; do
    (cd "$tree" && zig build drama3-generate -Dsm=121 -Dnvcc="$nvcc" -Doptimize=ReleaseFast)
done
(cd "$root" && zig build install fatbins kernel-contract-fatbins -Dsm=80,86,89,120,121 -Dnvcc="$nvcc" -Doptimize=ReleaseFast)
text='The quick brown fox jumps over the lazy dog.'
run() { # TREE MODEL NAME [ARGS...]: greedy codes into NAME.codes, stderr into NAME.log
    bin=$1/zig-out/bin/drama3-generate model=$2 name=$3
    shift 3
    "$bin" --model "$model" --text "$text" --greedy --max-frames 200 --out "$out/$name.codes" "$@" 2>"$out/$name.log"
    tail -n 1 "$out/$name.log" | grep -q '^RESULT drama3 OK'
}
run "$main_tree" "$fp8" fp8-main
run "$root" "$fp8" fp8-head
cmp "$out/fp8-main.codes" "$out/fp8-head.codes"
run "$root" "$compact" compact --codec "$codec" --replay 1
# a pruned checkpoint: an unpruned NVFP4 one keeps the original router in every layer and stops here
grep -q 'routed experts NVFP4 in 40 of 40 layers; compact routers in 40 of 40 layers' "$out/compact.log"
run "$root" "$compact" compact-batched --codec "$codec" --max-rows 64
cmp "$out/compact.codes" "$out/compact-batched.codes"
"$root/zig-out/bin/tf-cuda-test" drama3-router-compact 2>"$out/router.log"
"$root/zig-out/bin/tf-cuda-test" kernel-contract-limits "$root/zig-out/fatbin" >"$out/limits.log" 2>&1 || true
{
    echo "head $(git -C "$root" rev-parse HEAD) base $base"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    "$nvcc" --version | tail -n 1
    for f in "$fp8" "$compact" "$codec"; do echo "input $f: $(find "$f" -name '*.safetensors' -o -name '*.pth' | sort | xargs cat | sha256sum | cut -d' ' -f1)"; done
    echo "PASS fp8 greedy codes byte-identical to main: $(sha256sum "$out/fp8-head.codes" | cut -d' ' -f1)"
    echo "PASS compact $(grep -o 'routed experts held.*' "$out/compact.log")"
    echo "PASS compact decode and replay: $(tail -n 1 "$out/compact.log")"
    echo "PASS compact batched prefill (64 rows) codes equal one-row: $(sha256sum "$out/compact.codes" | cut -d' ' -f1)"
    grep '^PASS drama3-router-compact' "$out/router.log"
    grep '^RECORD drama3 d3_router_compact:' "$out/limits.log"
} >"$out/RECEIPT.txt"
cat "$out/RECEIPT.txt"
