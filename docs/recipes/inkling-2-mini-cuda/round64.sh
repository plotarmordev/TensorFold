#!/bin/sh
# Run on TOP through swarm gpu-run: the stored experts' contract (64-row decode entries and their controls), the
# checkpoint's 64-row items against the 16-row plan, and the expert items every MoE visit of one served 64-stream round
# holds. build.sh logs "build: N s"; the checks run under `timeout 580` and log "measure: N s".
set -eu
if [ "${1:-}" != measure ]; then
    start=$(date +%s)
    . docs/recipes/inkling-2-mini-cuda/build.sh
    echo "build: $(($(date +%s) - start)) s"
    export model out bin
    start=$(date +%s)
    status=0
    timeout --signal=INT --kill-after=20s 580 sh "$0" measure || status=$?
    echo "measure: $(($(date +%s) - start)) s"
    exit "$status"
fi
python3 -c 'import random; r = random.Random(4387); print(" ".join(str(r.randrange(1000, 199000)) for _ in range(64)))' >"$out/ids.txt"
"$bin/tf-cuda-test" stored-experts contract
"$bin/tf-cuda-test" stored-experts ckpt "$model"
"$bin/tf-cuda-test" inkling2-round64 "$model" "$out/ids.txt"
