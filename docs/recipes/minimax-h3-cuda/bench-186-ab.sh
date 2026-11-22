#!/bin/sh
# Card 186's same-job A/B: --routed-steps (FastH3's tile top-k in the last steps) against dense, on this commit.
# Each pass renders the pilot fox cold (episode 1's gate prompt) and three prompts no earlier card used, warm, with
# one resident server a pass: dense, routed 2 and routed 3 with CUDA graphs (the whole picture), then dense and
# routed 2 under H3_PROFILE (the attention and routing stages). Each routed picture is gated against the dense one of
# the same prompt with gate.py (TOP has no numpy or PIL). Results: /work/out/h3-186-* and the stage lines on stdout.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-186-ab.sh
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot/h3-image-pilot
p=/work/h3-prompts-186
mkdir -p $p
cat >$p/lighthouse.txt <<'EOF'
A lighthouse keeper's cottage on a basalt cliff during a winter storm, whitewashed walls streaked with rain,
a lit window, waves exploding against black rocks below, the beam cutting through sleet, dramatic overcast
sky, cinematic wide shot, high detail.
EOF
cat >$p/workshop.txt <<'EOF'
A watchmaker's workbench under a magnifying lamp: tiny brass gears, jewel bearings and tweezers on green felt,
an open pocket watch movement, a loupe and a notebook of hand-drawn diagrams, shallow depth of field, macro
photograph, warm tungsten light.
EOF
cat >$p/glacier.txt <<'EOF'
An ice cave inside a blue glacier, translucent walls with trapped air bubbles and layered sediment, a lone
climber in a red jacket holding a headlamp, meltwater stream on the floor, cold daylight filtering through,
ultra-wide lens, photorealistic.
EOF
prompts="$pilot/prompt.txt $p/lighthouse.txt $p/workshop.txt $p/glacier.txt"
status=0
bench() { # bench OUT ROUTED PROFILE: the four runs into /work/out/OUT
    H3_BENCH_OUT=/work/out/$1 H3_ROUTED_STEPS=$2 H3_PROFILE=$3 H3_BENCH_SKIP_COMPARE=skip H3_BENCH_PROMPTS="$prompts" \
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
bench h3-186-dense "" ""
bench h3-186-routed2 2 ""
bench h3-186-routed3 3 ""
bench h3-186-dense-profile "" 1
bench h3-186-routed2-profile 2 1
for side in dense routed2 routed3 dense-profile routed2-profile; do
    echo "=== $side ==="
    cat /work/out/h3-186-$side/selftests || true
    stages /work/out/h3-186-$side
done
echo "=== gates: each routed picture against the dense one of its prompt ==="
for side in routed2 routed3 routed2-profile; do
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