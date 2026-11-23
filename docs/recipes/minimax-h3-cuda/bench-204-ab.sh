#!/bin/sh
# Card 204's same-job picture gate (job clones carry no history to check main out from): bench-script.sh on this
# commit, then bench-204-base.patch reverse-applied (main's TMA product: no edge tile) and the same run again — the
# pilot prompt and one unseen prompt at 2048², H3_PROFILE off. The served projections' N are whole 128-column
# tiles, so every latent and PNG must be main's bytes; gate.py's cosine and PSNR against main are quoted too.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-204-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt"
status=0
bench() { # bench OUT: the two runs into /work/out/OUT
    H3_BENCH_OUT=/work/out/$1 H3_PROFILE= H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" \
        sh $here/bench-script.sh || status=$?
}
echo "=== card: this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
bench h3-204-card
echo "=== main: the card's files at main (patch -R) ==="
git -c safe.directory='*' apply -R $here/bench-204-base.patch
bench h3-204-main
git -c safe.directory='*' apply $here/bench-204-base.patch
echo "=== pictures, card against main ==="
for run in 1 2; do
    c=/work/out/h3-204-card/2048-$run
    m=/work/out/h3-204-main/2048-$run
    python3 $here/gate.py "$m" "$c" | tee /work/out/h3-204-gate-$run.txt
    grep -q ": PASS$" /work/out/h3-204-gate-$run.txt || status=1
    for f in latent.f32 png; do
        cmp -s "$c.$f" "$m.$f" && echo "2048-$run.$f identical" || { echo "2048-$run.$f DIFFERS"; status=1; }
    done
done
exit "$status"
