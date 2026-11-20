#!/bin/sh
# Card 144's same-job A/B (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-144-base.patch reverse-applied — the card's decoder files back to main's — and
# bench-script.sh again. The decoder doesn't depend on the prompt, so both phases render the canonical
# (seen) pilot prompt: run 1 cold, runs 2-4 warm kept-states hits, H3_PROFILE on.
# This commit's phase also runs bench-script.sh's same-latent `reference` compare (the decoder's
# full-chain bound); run 1's latent and PNG face the episode-1 gates against the Mac references in-job;
# every run's latent must be byte-identical between the phases (the transformer is untouched).
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-144-ab.sh
set -eu
patch_file=docs/recipes/minimax-h3-cuda/bench-144-base.patch
status=0
stages() { # every run's H3_PROFILE decoder stage lines
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run decoder stages"
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.decoder.log" || true
    done
}
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
H3_PROFILE=1 sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
stages /work/out/h3-image-bench
grep -h "relative\|PSNR" /work/out/h3-image-bench/2048-reference*.log || true
echo "=== episode-1 gates: run 1's latent and PNG against the Mac references ==="
python3 - /assets/drama3-pilot/h3-image-runs/34/t2i-2048.latent.f32 /work/out/h3-image-bench/2048-1.latent.f32 \
    /assets/drama3-pilot/ep1-mac-fast/2048-warm1.png /work/out/h3-image-bench/2048-1.png <<'PY' || status=1
import array, math, sys
from PIL import Image
a, b = (array.array("f", open(p, "rb").read()) for p in sys.argv[1:3])
cosine = sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
pa, pb = (Image.open(p).convert("RGB").tobytes() for p in sys.argv[3:5])
mae = sum(abs(x - y) for x, y in zip(pa, pb)) / len(pa)
mse = sum((x - y) ** 2 for x, y in zip(pa, pb)) / len(pa)
psnr = 10 * math.log10(255 * 255 / mse)
ok = cosine >= 0.99 and psnr >= 28.2 and mae <= 4.1
print(f"episode-1 gates {'PASS' if ok else 'FAIL'}: cosine {cosine:.4f}, PSNR {psnr:.2f} dB, mean {mae:.2f}")
sys.exit(0 if ok else 1)
PY
echo "=== before: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R "$patch_file"
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_PROFILE=1 H3_BENCH_SKIP_COMPARE=1 \
    sh docs/recipes/minimax-h3-cuda/bench-script.sh || status=$?
git -c safe.directory='*' apply "$patch_file"
stages /work/out/h3-image-bench-base
echo "=== latents, this commit against main ==="
for run in 1 2 3 4; do
    if cmp -s /work/out/h3-image-bench/2048-$run.latent.f32 /work/out/h3-image-bench-base/2048-$run.latent.f32; then
        echo "2048-$run.latent.f32 identical"
    else
        echo "2048-$run.latent.f32 DIFFERENT"
        status=1
    fi
done
exit "$status"
