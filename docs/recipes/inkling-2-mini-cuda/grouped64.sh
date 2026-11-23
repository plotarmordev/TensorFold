#!/bin/sh
# Run on TOP through swarm gpu-run: same-prompt and distinct-prompt 64-stream greedy decode.
set -eu
. docs/recipes/inkling-2-mini-cuda/setup.sh
mem() {
    awk -v label="$1" '/^MemTotal:/ {total = $2} /^MemAvailable:/ {avail = $2} END {used = (total-avail)/1048576; printf "%s host used %.1f GiB, available %.1f GiB\n", label, used, avail; if (used >= 128) exit 1}' /proc/meminfo
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
}
mem before
"/work/sm121/native/bin/tensorfold-native" serve "$model" --backend cuda --host 127.0.0.1 --port 8188 \
    --name inkling-2-mini --parallel 64 --context 4096 --no-update-check >"$out/server-grouped64.log" 2>&1 &
server=$!
trap 'kill -INT "$server" 2>/dev/null || :; wait "$server" 2>/dev/null || :' EXIT
python3 - "$server" <<'PY'
import sys, time, urllib.request
for _ in range(900):
    try:
        urllib.request.urlopen('http://127.0.0.1:8188/v1/models', timeout=5)
        break
    except OSError:
        try:
            state = open(f'/proc/{sys.argv[1]}/stat').read().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            state = 'Z'
        if state == 'Z':
            raise SystemExit('server exited before answering')
        time.sleep(1)
else:
    raise SystemExit('server did not answer in 15 min')
PY
mem loaded
python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels 64 --tokens 128 --reps 3 \
    --temperatures 0 --alone --label grouped64-same --output "$out/grouped64-same.json"
python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels 64 --tokens 128 --reps 3 \
    --temperatures 0 --prompts tools/bench_prompts.json --alone --label grouped64-distinct \
    --output "$out/grouped64-distinct.json"
mem after
python3 - "$out" <<'PY'
import json, statistics, sys
for kind in ('same', 'distinct'):
    cells = json.load(open(f'{sys.argv[1]}/grouped64-{kind}.json'))['cells']
    for cell in cells:
        same = cell.get('alone', {})
        if cell['failed'] or same.get('unequal') or same.get('failed') or same.get('equal') != 64 * 3:
            raise SystemExit(f'{kind} {cell["prompt"]}: failed={cell["failed"]}, solo comparison={same}')
        print(f'RESULT grouped64 {kind} {cell["prompt"]}: median of 3 total decode '
              f'{statistics.median(r["aggregate_tps"] for r in cell["reps"]):.1f} tok/s; '
              f'failed=0, solo-equal={same["equal"]}')
PY
