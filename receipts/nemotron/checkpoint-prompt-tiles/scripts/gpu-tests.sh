#!/bin/sh
# tf-cuda-test on the tip (checkpoint, glue, modelopt, kernel-contract, limits of the sm_80..121 fatbins) and on the
# planted mutant (checkpoint, glue, kernel-contract); one GPU hold a binary. Logs in <work>/gpu.
set -u
here=<work>
out=$here/gpu
mkdir -p "$out"
export TF_HEAT_HIGH=85 TF_HEAT_LOW=75
banner() { echo "source $(head -1 "$here/$1/commit") ($2), tf-cuda-test sha256 $(sha256sum "$here/$1/tf-cuda-test" | cut -d' ' -f1)"; }
run() {
    dir=$1 what=$2
    shift 2
    for cmd in "$@"; do
        log=$out/$cmd.$dir.log
        { banner "$dir" "$what"; timeout 8m "$here/$dir/tf-cuda-test" $cmd 2>&1; echo "exit=$?"; } >"$log"
    done
    nvidia-smi --query-compute-apps=pid --format=csv,noheader
}
case $1 in
tip) run tip tip checkpoint glue modelopt kernel-contract
    { banner tip "tip, sm_80/86/89/120/121 fatbins of the same checkout"; timeout 5m "$here/tip/tf-cuda-test" kernel-contract-limits "$here/sm5/fatbin" 2>&1; echo "exit=$?"; } >"$out/kernel-contract-limits.tip.log" ;;
mut) run mut "tip + mutant.diff" checkpoint glue kernel-contract ;;
esac
