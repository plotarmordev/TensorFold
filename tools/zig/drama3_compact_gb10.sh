#!/bin/sh
# Drama 3's compact NVFP4 experts on one GB10, the checks the host tests cannot make: the compact checkpoint loads
# as stored and decodes (eager, graphs and interleaved state agree under --replay), the stock FP8 checkpoint's
# greedy codes are byte-identical to origin/main's, and d3_router_compact's sm_121 resources for its contract record.
# Writes RECEIPT.txt and every log into RECEIPT_DIR.
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
    (cd "$tree" && zig build drama3-generate -Dgpu=spark -Dnvcc="$nvcc" -Doptimize=ReleaseFast)
done
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
grep -q 'routed experts NVFP4 in 40 of 40 layers' "$out/compact.log"
"$nvcc" -cubin -arch=sm_121 -O3 -Xptxas -v -I "$root/zig/kernels/cuda" -o "$out/drama3.cubin" "$root/zig/kernels/cuda/drama3.cu" 2>"$out/ptxas.log"
usage=$(grep -A2 'd3_router_compact' "$out/ptxas.log" | grep -o 'Used .*')
{
    echo "head $(git -C "$root" rev-parse HEAD) base $base"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo "PASS fp8 greedy codes byte-identical to main: $(sha256sum "$out/fp8-head.codes" | cut -d' ' -f1)"
    echo "PASS compact $(grep -o 'routed experts held.*' "$out/compact.log")"
    echo "PASS compact decode and replay: $(tail -n 1 "$out/compact.log")"
    echo "RECORD d3_router_compact sm_121: $usage"
} >"$out/RECEIPT.txt"
cat "$out/RECEIPT.txt"
