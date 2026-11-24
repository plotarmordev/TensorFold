#!/bin/sh
# Run on TOP through swarm gpu-run: the stored experts' contract (64-row decode entries and their controls), the
# checkpoint's 64-row items against the 16-row plan, and the expert items of one served 64-stream round.
set -eu
. docs/recipes/inkling-2-mini-cuda/build.sh
python3 -c 'import random; r = random.Random(4387); print(" ".join(str(r.randrange(1000, 199000)) for _ in range(64)))' >"$out/ids.txt"
"$bin/tf-cuda-test" stored-experts contract
"$bin/tf-cuda-test" stored-experts ckpt "$model"
"$bin/tf-cuda-test" inkling2-round64 "$model" "$out/ids.txt"
