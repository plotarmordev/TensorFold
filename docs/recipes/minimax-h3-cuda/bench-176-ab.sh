#!/bin/sh
# Card 176's job: the int8 contract (gates 1-4 on the per-block TMA product and h3_quantize_blocks, gate 5's
# live sm_121 limits), then a same-job A/B of the pictures (job clones carry no history to check main out
# from): bench-script.sh on this commit (out and fc2 int8 at a scale per 128 of K), then
# bench-176-base.patch reverse-applied (out and fc2 back to bf16, main's default) and the same runs, then
# this commit with --bf16 as the picture reference. Each phase renders the pilot prompt cold and three
# prompts no earlier job used warm, H3_PROFILE on for the stage times. The picture gates (latent cosine
# >= 0.99, PSNR >= 28.2 dB, mean diff <= 4.1) run here on every warm picture against bf16's, stdlib only:
# TOP has no PIL and the PNGs are filter-0 RGB.
# Results: /work/out/h3-176-{card,main,bf16} and GATE lines on stdout.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-176-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot/h3-image-pilot
p=/work/h3-prompts-176
mkdir -p $p
cat >$p/glacier.txt <<'EOF'
A blue ice cave inside a glacier, light filtering through layered translucent walls, a lone climber in a red
jacket holding a headlamp, frozen bubbles and cracks catching the glow, wide angle, crisp detail.
EOF
cat >$p/bakery.txt <<'EOF'
A village bakery before dawn: flour on a worn wooden counter, rows of crusty loaves cooling on racks, a baker
shaping dough under a single hanging bulb, steam rising from the oven door, warm tones, film grain.
EOF
cat >$p/orchard.txt <<'EOF'
An apple orchard in late autumn after rain, wet grass and fallen red apples, a wooden ladder against a gnarled
tree, low golden sun through thin fog, a wicker basket half full, shallow depth of field.
EOF
prompts="$pilot/prompt.txt $p/glacier.txt $p/bakery.txt $p/orchard.txt"
status=0
echo "=== contract: h3 int8 gates 1-4 and live limits ==="
H3_BUILD=contract H3_CONTRACT=int8 sh $here/cuda-build.sh || status=$?
bench() { # bench OUT [LINEAR]: the four runs into /work/out/OUT
    H3_BENCH_OUT=/work/out/$1 H3_PROFILE=1 H3_LINEAR=${2:-} H3_BENCH_SKIP_COMPARE=skip H3_BENCH_PROMPTS="$prompts" \
        sh $here/bench-script.sh || status=$?
}
stages() { # every run's wall and the transformer's H3_PROFILE block stages
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run"
        awk 'NR == 1 { s = $1 } END { printf "wall %.2f s\n", $1 - s }' "$1/2048-$run.time"
        grep -hE "^transformer: " "$1/2048-$run.image.log" || true
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.image.log" || true
    done
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
bench h3-176-card
echo "=== main: out and fc2 bf16 (patch -R) ==="
git -c safe.directory='*' apply -R $here/bench-176-base.patch
bench h3-176-main
git -c safe.directory='*' apply $here/bench-176-base.patch
echo "=== bf16: this commit, --bf16 ==="
bench h3-176-bf16 bf16
for side in card main bf16; do
    echo "=== $side ==="
    stages /work/out/h3-176-$side
done
echo "=== picture gates against bf16 ==="
for side in card main; do
    python3 - /work/out/h3-176-$side /work/out/h3-176-bf16 $side <<'EOF' || status=1
import math, struct, sys, zlib
from array import array

def rgb(path):
    data = open(path, "rb").read()
    width, height = struct.unpack(">II", data[16:24])
    raw, at = b"", 8
    while at < len(data):
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        if kind == b"IDAT":
            raw += data[at + 8:at + 8 + size]
        at += 12 + size
    rows = zlib.decompress(raw)
    stride = width * 3 + 1
    assert all(rows[y * stride] == 0 for y in range(height)), "filter 0 rows only"
    return b"".join(rows[y * stride + 1:(y + 1) * stride] for y in range(height))

side, ref, name = sys.argv[1], sys.argv[2], sys.argv[3]
failed = False
for run in (2, 3, 4):
    a, b = array("f"), array("f")
    a.frombytes(open(f"{ref}/2048-{run}.latent.f32", "rb").read())
    b.frombytes(open(f"{side}/2048-{run}.latent.f32", "rb").read())
    cosine = sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    p, q = rgb(f"{ref}/2048-{run}.png"), rgb(f"{side}/2048-{run}.png")
    total = squares = 0
    for x, y in zip(p, q):
        d = x - y
        total += abs(d)
        squares += d * d
    mean, mse = total / len(p), squares / len(p)
    psnr = 10 * math.log10(255 * 255 / mse) if mse else math.inf
    ok = cosine >= 0.99 and psnr >= 28.2 and mean <= 4.1
    failed = failed or not ok
    print(f"GATE {name} 2048-{run} vs bf16: latent cosine {cosine:.4f}, PSNR {psnr:.2f} dB, mean diff {mean:.2f}: {'PASS' if ok else 'FAIL'}")
sys.exit(1 if failed and name == "card" else 0)
EOF
done
exit "$status"
