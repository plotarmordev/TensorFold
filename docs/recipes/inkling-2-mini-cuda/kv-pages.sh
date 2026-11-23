#!/bin/sh
# Inkling 2 Mini's K/V pages by TMA bulk copies on TOP's GB10:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/kv-pages.sh
# After setup.sh's build and gates: the attention contract on the candidate and on a control built from the same
# tree with TF_INK2_KV_LOADS=1 (the per-thread loads the bulk copies replace; the unchanged prompt kernel must
# equal the fold's bytes in both), then fresh servers control, candidate, control at 64 streams, 1K and 8K
# context, greedy, 128 tokens, 3 reps, each stream its own prompt of about 860 or 8,000 tokens. Achieved K/V
# bandwidth is the bytes a decoded token's attention reads by the cache layout (40 sliding visits of at most 1,024
# keys, 8 global visits of the whole context; 8 key heads, K and V 512 B a key) times the measured decode rate,
# not a counter. Any failed request stops the job. The compile records print last.
set -eu
. docs/recipes/inkling-2-mini-cuda/setup.sh
mkdir -p /work/attn-usage
for arm in candidate control; do
    define=$([ "$arm" = control ] && echo "-DTF_INK2_KV_LOADS=1" || echo -O3)
    /usr/local/cuda/bin/nvcc -fatbin -std=c++20 --expt-relaxed-constexpr -O3 $define -I zig/kernels/cuda \
        -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 \
        -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 -o "/work/attn-usage/$arm.fatbin" \
        zig/kernels/cuda/inkling2/attention.cu
    /usr/local/cuda/bin/cuobjdump --dump-resource-usage "/work/attn-usage/$arm.fatbin" | awk -v arm="$arm" '
        /arch = / { arch = $3 } /Function tf_ink2_attn/ { fn = $2 } /REG:|SHARED:/ && fn { print "RESULT", arm, arch, fn, $0 }'
done
"$bin/tf-cuda-test" kernel-contract "inkling2 attn"
NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS -DTF_INK2_KV_LOADS=1" ZIG_LOCAL_CACHE_DIR=/work/zig-control-cache \
    zig build install native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/control
control=/work/control
cmp "$control/native/bin/tensorfold-native" /work/sm121/native/bin/tensorfold-native >/dev/null && {
    echo "the control build repeated the candidate's binary" >&2; exit 1;
}
"$control/bin/tf-cuda-test" kernel-contract "inkling2 attn"
"$bin/tf-cuda-test" kernel-contract "inkling2_attn"
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
for context in 1024 8192; do
    for arm in control-1 candidate control-2; do
        arm "$context" "$arm" "$([ "$arm" = candidate ] && echo /work/sm121/native/bin/tensorfold-native || echo "$control/native/bin/tensorfold-native")"
    done
    summary "$context"
done
