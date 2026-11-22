#!/bin/sh
# Card 186's same-job A/B: --routed-steps 2 (FastH3's tile top-k in the last two steps) against dense, on this commit.
# Each pass renders the pilot fox cold (episode 1's gate prompt) and three prompts no earlier card used, warm, with
# one resident server a pass: dense and routed with CUDA graphs (the whole picture), then both under H3_PROFILE (the
# attention and routing stages). Each routed picture is gated against the dense one of the same prompt with gate.py
# (TOP has no numpy or PIL). Results: /work/out/h3-186-* and the stage lines on stdout.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-186-ab.sh
# Run at d5968c7e (job top-e1cd6974b1da): routed 2 kept opt-in, the gates hold; numbers in minimax-h3-cuda.md.
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot/h3-image-pilot
p=/work/h3-prompts-186
mkdir -p $p
cat >$p/observatory.txt <<'EOF'
A mountaintop observatory under the Milky Way, its white dome open and the telescope pointed at the sky, a thin
crescent moon over snowy ridges, red safety lights along a gravel path, long exposure, astrophotography.
EOF
cat >$p/bakery.txt <<'EOF'
A small village bakery before dawn: a baker in a flour-dusted apron sliding loaves out of a brick oven, racks
of croissants and rye bread, steam in the warm light, tiled walls and a chalkboard menu, documentary photograph.
EOF
cat >$p/reef.txt <<'EOF'
A shallow coral reef seen from just below the surface, sunlight rippling over brain coral and sea fans, a
school of yellow fish turning at once, a green sea turtle gliding past, crystal-clear turquoise water,
underwater wide-angle photograph, vivid natural colour.
EOF
prompts="$pilot/prompt.txt $p/observatory.txt $p/bakery.txt $p/reef.txt"
status=0
bench() { # bench SIDE ROUTED PROFILE: the four runs into /work/out/h3-186-SIDE
    H3_BENCH_OUT=/work/out/h3-186-$1 H3_ROUTED_STEPS=$2 H3_PROFILE=$3 H3_BENCH_SKIP_COMPARE=skip H3_BENCH_PROMPTS="$prompts" \
        sh $here/bench-script.sh || status=$?
}
stages() { # every run's wall, transformer time, step times and (profiled) block stages
    for run in 1 2 3 4; do
        echo "--- $1 2048-$run"
        tr '\n' ' ' <"$1/2048-$run.time" | awk '{printf "wall %.2f s\n", $2 - $1}'
        grep -hE "^transformer: |^step [0-9]" "$1/2048-$run.image.log" || true
        grep -E '^  [a-z ]+: [0-9.]+ s$' "$1/2048-$run.image.log" || true
    done
}
echo "=== card $(git -c safe.directory='*' rev-parse --short HEAD) ==="
bench dense "" ""
bench routed 2 ""
bench dense-profile "" 1
bench routed-profile 2 1
for side in dense routed dense-profile routed-profile; do
    echo "=== $side ==="
    cat /work/out/h3-186-$side/selftests || true
    stages /work/out/h3-186-$side
done
echo "=== gates: each routed picture against the dense one of its prompt ==="
for side in routed routed-profile; do
    for run in 1 2 3 4; do
        python3 $here/gate.py /work/out/h3-186-dense/2048-$run /work/out/h3-186-$side/2048-$run
    done
done
echo "=== dense: graphs against the profiled pass (the same picture), and the fox's latent (main's is 8d067d75…) ==="
for run in 1 2 3 4; do
    cmp -s /work/out/h3-186-dense/2048-$run.latent.f32 /work/out/h3-186-dense-profile/2048-$run.latent.f32 \
        && echo "2048-$run identical" || echo "2048-$run differs"
done
sha256sum /work/out/h3-186-dense/2048-1.latent.f32
exit "$status"
