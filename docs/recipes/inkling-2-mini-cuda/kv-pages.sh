#!/bin/sh
# Inkling 2 Mini's K/V pages by TMA bulk copies on TOP's GB10:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/kv-pages.sh (1K; kv-pages-8k.sh: 8K)
# The whole job runs under `timeout 600`, so setup.sh only unpacks the toolchain (its build and gates are other
# receipts'). The candidate and a control built from the same tree and cache with -Dinkling2-kv-loads=true (the
# per-thread loads the bulk copies replace; only that kernel image and the executables rebuild) dump the attention
# contract's outputs at every served form: 1 row and multi-row verify, keys to 1M wrapping the sliding (2,048) and
# global (4,096) rings, page tails (1,030, 16,390, 20,000, 300,000 keys). Both builds' bytes must match. Then fresh
# servers control, candidate, control at 64 streams, one context a job, greedy, 128 tokens, 3 reps, each stream its
# own prompt of about 860 or 8,000 tokens. Achieved K/V is derived from the cache layout and the measured decode rate,
# not a counter. Setup, builds and the byte comparison log "setup: N s"; the arms log "measure: N s" and stop the
# job on any failed request.
set -eu
[ -n "${KV_PAGES_BOUNDED:-}" ] || KV_PAGES_BOUNDED=1 exec timeout -k 5 600 sh "$0"
control=/work/control
start=$(date +%s)
INKLING_ENV_ONLY=1 . docs/recipes/inkling-2-mini-cuda/setup.sh
build() {
    zig build install native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast -Dcuda-test-dump=true \
        --prefix "$@"
}
build /work/sm121
build "$control" -Dinkling2-kv-loads=true
cmp "$control/native/bin/tensorfold-native" /work/sm121/native/bin/tensorfold-native >/dev/null && {
    echo "the control build repeated the candidate's binary" >&2; exit 1;
}
contract() {
    TF_CUDA_DUMP=1 "$1/bin/tf-cuda-test" kernel-contract "inkling2 attn" >"$1.log" 2>&1 || { cat "$1.log"; exit 1; }
    grep -v '^DUMP ' "$1.log"
    grep '^DUMP ' "$1.log" >"$1.dump" || :
}
contract /work/sm121
contract "$control"
[ -s "$control.dump" ] && cmp -s /work/sm121.dump "$control.dump" || {
    echo "attention outputs differ between the bulk copies and the per-thread loads" >&2; exit 1;
}
echo "PASS attention outputs: $(wc -l <"$control.dump") served forms byte-identical, bulk copies and per-thread loads"
echo "setup: $(($(date +%s) - start)) s"
start=$(date +%s)
prompts() {
    python3 - "$out" "$1" "$2" <<'PY'
import json, random, sys, urllib.request
out, context, target = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
words = ('amber cedar violet river copper willow glass lantern maple ocean honey silver coral orbit meadow '
         'ivory orchard velvet sunset cobalt pebble forest').split()
def text(count, seed):
    return ' '.join(random.Random(seed).choices(words, k=count))
def tokens(prompt):
    body = {'model': 'inkling-2-mini', 'prompt': prompt, 'max_tokens': 1, 'temperature': 0}
    req = urllib.request.Request('http://127.0.0.1:8188/v1/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))['usage']['prompt_tokens']
count = round(400 * target / tokens(text(400, 240)))
with open(f'{out}/prompts-{context}.json', 'w') as f:
    json.dump([{'name': f'stream-{i}', 'kind': 'completion', 'prompt': text(count, 240 + i)} for i in range(64)], f)
PY
}
arm() {
    context=$1 name=$2 binary=$3
    "$binary" serve "$model" --backend cuda --host 127.0.0.1 --port 8188 --name inkling-2-mini --parallel 64 \
        --context "$context" --no-update-check >"$out/server-$context-$name.log" 2>&1 &
    server=$!
    trap 'kill -INT "$server" 2>/dev/null || :' EXIT
    python3 - "$server" <<'PY'
import sys, time, urllib.request
for _ in range(900):
    try:
        urllib.request.urlopen('http://127.0.0.1:8188/v1/models', timeout=5)
        raise SystemExit(0)
    except OSError:
        # the server is the shell's child, not ours: a zombie or a gone pid has exited
        try:
            state = open(f'/proc/{sys.argv[1]}/stat').read().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            state = 'Z'
        if state == 'Z':
            raise SystemExit('the server exited before answering')
        time.sleep(1)
raise SystemExit('server did not answer /v1/models in 15 min')
PY
    [ -f "$out/prompts-$context.json" ] || prompts "$context" "$([ "$context" = 1024 ] && echo 860 || echo 7800)"
    python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels 64 --tokens 128 --reps 3 \
        --temperatures 0 --prompts "$out/prompts-$context.json" --label "$context-$name" \
        --output "$out/$context-$name.json" >/dev/null
    kill -INT "$server"
    wait "$server" || :
    trap - EXIT
}
summary() {
    python3 - "$out" "$1" <<'PY'
import json, statistics, sys
out, context = sys.argv[1], int(sys.argv[2])
cells = {arm: json.load(open(f'{out}/{context}-{arm}.json'))['cells'][0]
         for arm in ('control-1', 'candidate', 'control-2')}
for rep in range(3):
    hashes = [[q['token_sha256'] for q in cells[arm]['reps'][rep]['evidence']] for arm in cells]
    if any(h is None for row in hashes for h in row) or len(hashes[0]) != 64 or not (hashes[0] == hashes[1] == hashes[2]):
        raise SystemExit(f'{context} rep {rep + 1}: control/candidate token bytes differ')
print(f'PASS {context}: 192/192 stream replies byte-identical across control/candidate/control')
for arm in ('control-1', 'candidate', 'control-2'):
    cell = cells[arm]
    if cell['failed'] or cell['unmeasured'] or not cell['timing_valid']:
        raise SystemExit(f'{context} {arm}: {cell["failed"]} failed, {cell["unmeasured"]} unmeasured requests')
    reps = cell['reps']
    p = statistics.median(q['prompt_tokens'] for r in reps for q in r['evidence'])
    keys = statistics.mean(40 * min(p + t, 1024) + 8 * (p + t) for t in range(128))
    decode = statistics.median(r['aggregate_tps'] for r in reps)
    print(f'RESULT inkling2 mini {context} context {arm}: prompt {p:.0f} tokens, K/V {keys * 4096 / 1e6:.0f} MB a token, '
          f'achieved K/V {decode * keys * 4096 / 1e9:.0f} GB/s, 64 streams {decode:.1f} tok/s, '
          f'end to end {statistics.median(r["end_to_end_tps"] for r in reps):.1f} tok/s (medians of 3)')
PY
}
context=${KV_PAGES_CONTEXT:-1024}
for arm in control-1 candidate control-2; do
    arm "$context" "$arm" "$([ "$arm" = candidate ] && echo /work/sm121/native/bin/tensorfold-native || echo "$control/native/bin/tensorfold-native")"
done
summary "$context"
echo "measure: $(($(date +%s) - start)) s"
