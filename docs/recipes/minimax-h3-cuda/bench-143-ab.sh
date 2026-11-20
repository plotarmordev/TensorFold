#!/bin/sh
# Card 143's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit — the prompt encoder resident in int8, each run's prompt from H3_BENCH_PROMPTS so the three
# warm 2048 runs are unseen — then bench-143-base.patch reverse-applied (the card's files back to
# main's bf16-only encoder) and bench-script.sh again: the baseline is the same job's main code with
# this commit's bench script, model paths and prompt list.
# Results: /work/out/h3-image-bench (int8) and /work/out/h3-image-bench-base (main's bf16 encoder);
# the states gate compares each run's text tensor between the two phases (cosine >= 0.999, ids and
# tags byte-equal), and cuobjdump reports the two new kernels' measured sm_121 usage.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-143-ab.sh"
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-143-base.patch
pilot=/assets/drama3-pilot
prompts="$pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt $here/prompts/meadow.txt $here/prompts/steam.txt"
status=0
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD), resident int8 encoder ==="
H3_PROFILE=1 H3_BENCH_PROMPTS="$prompts" sh $here/bench-script.sh || status=$?
echo "=== before: the card's files at main (patch -R), bf16 encoder ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_BENCH_SKIP_COMPARE=1 H3_PROFILE=1 H3_BENCH_PROMPTS="$prompts" \
    sh $here/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"

echo "=== prompt states: int8 runs against the same job's main-bf16 runs, cosine >= 0.999 ==="
python3 - "$status" <<'PY' || status=$?
import json, math, struct, sys


def tensor(path, name):
    with open(path, 'rb') as f:
        raw = f.read()
    n = struct.unpack('<Q', raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    start, end = head[name]['data_offsets']
    return raw[8 + n + start:8 + n + end]


def bf16_cosine(a, b):
    dot = squares_a = squares_b = 0.0
    for x, y in zip(struct.iter_unpack('<H', a), struct.iter_unpack('<H', b)):
        u = struct.unpack('<f', struct.pack('<I', x[0] << 16))[0]
        v = struct.unpack('<f', struct.pack('<I', y[0] << 16))[0]
        dot += u * v
        squares_a += u * u
        squares_b += v * v
    return dot / math.sqrt(squares_a * squares_b)


status = 0
for run in range(1, 5):
    a = '/work/out/h3-image-bench/2048-%d.safetensors' % run
    b = '/work/out/h3-image-bench-base/2048-%d.safetensors' % run
    cosine = bf16_cosine(tensor(a, 'text'), tensor(b, 'text'))
    same = tensor(a, 'ids') == tensor(b, 'ids') and tensor(a, 'text_tags') == tensor(b, 'text_tags')
    ok = cosine >= 0.999 and same
    print('2048-%d: text cosine %.6f, ids and tags byte-equal: %s (%s)' % (run, cosine, same, 'PASS' if ok else 'FAIL'))
    if not ok:
        status = 1
sys.exit(status or int(sys.argv[1]))
PY
exit "$status"
