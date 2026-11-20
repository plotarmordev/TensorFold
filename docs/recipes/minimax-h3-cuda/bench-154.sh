#!/bin/sh
# Card 154's same-job check: bench-script.sh on this commit with the pilot prompt, then five unseen prompts of
# 1 to 300+ tokens, each a 2048 picture through the resident servers, H3_PROMPT_CHECK=1 checking every encode's
# refiner rows against the BF16 host reference (cosine >= 0.999); then bench-154-base.patch reverse-applied (the
# products back to main's) and the pilot prompt alone again: its latent must be byte-equal to this commit's.
# Results: /work/out/h3-image-bench (this commit) and /work/out/h3-image-bench-base (main's products).
# ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-154.sh"
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot/h3-image-pilot
p=/work/h3-prompts-154
mkdir -p $p
echo cat >$p/one.txt
cat >$p/lighthouse.txt <<'EOF'
A weathered lighthouse keeper's cottage on a black basalt shore at blue hour, warm lamplight in two
windows, sea stack silhouettes behind low fog, wet stone reflections, muted teal and amber palette,
photorealistic, 35mm.
EOF
cat >$p/workshop.txt <<'EOF'
A cluttered violin maker's workshop in late afternoon: spruce tops clamped on the bench, curls of maple
shavings on the floor, rows of unvarnished scrolls hanging from a rail, a single window throwing a long
shaft of dusty gold light across jars of varnish, chisels and calipers, an old radio on the shelf,
photographed on medium format film.
EOF
cat >$p/glacier.txt <<'EOF'
An aerial photograph of a glacier tongue meeting a milky turquoise lake at dawn. Crevasses run in
long blue arcs across the ice; a thin line of moraine, dark and gravelly, curves along the left edge
where the ice has pulled back. Small icebergs float near the calving front, each casting a faint
shadow on the still water. The far shore is a slope of scree and patchy green moss, and a low bank of
cloud sits on the ridge above it, lit pink from below by the rising sun. Shot from a light aircraft,
crisp detail, natural colour, no people, high dynamic range.
EOF
cat >$p/station.txt <<'EOF'
A wide interior view of a grand nineteenth-century railway station on a winter morning. The roof is a
vast arch of wrought iron and glass, its panes frosted at the edges, and pale sunlight falls through it
in slanted beams that catch the steam drifting up from two waiting locomotives. One engine is deep green
with polished brass fittings, the other black and soot-streaked, and both stand at long stone platforms
lined with cast-iron lamp posts and wooden benches. Travellers in heavy wool coats and scarves move in
small groups: a porter pushes a trolley stacked with leather trunks, a woman in a red hat checks a paper
ticket, two children press their faces to a train window, and a conductor with a pocket watch looks up
at the great clock hanging from the central girder. A newspaper kiosk glows with warm yellow light near
the entrance, its racks full of folded papers and postcards. Pigeons perch along the iron beams and a few
flutter down toward crumbs on the platform. The floor is patterned tile, wet in places and reflecting
the lamps and the steam. In the far distance, beyond the end of the roof, snow-covered rooftops and a
church spire are visible under a cold blue sky. The scene is photographed with a wide lens from a raised
balcony, with rich detail, soft atmospheric haze, natural colour, gentle film grain and a calm, quiet
mood, every face small and turned away from the camera.
EOF
status=0
echo "=== after: this commit $(git -c safe.directory='*' rev-parse --short HEAD), prompt states checked against the host reference ==="
H3_PROMPT_CHECK=1 H3_BENCH_PROMPTS="$pilot/prompt.txt $p/one.txt $p/lighthouse.txt $p/workshop.txt $p/glacier.txt $p/station.txt" \
    sh $here/bench-script.sh || status=$?
for run in 1 2 3 4 5 6; do
    log=/work/out/h3-image-bench/2048-$run.prompt.log
    [ -f "$log" ] && grep -E "tokens;|host reference" "$log" | sed "s|^|2048-$run: |"
done
echo "=== before: main's products (patch -R), the pilot prompt alone ==="
git -c safe.directory='*' apply -R $here/bench-154-base.patch
H3_BENCH_OUT=/work/out/h3-image-bench-base H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$pilot/prompt.txt" \
    sh $here/bench-script.sh || status=$?
git -c safe.directory='*' apply $here/bench-154-base.patch
echo "=== the canonical prompt's latent, this commit against main ==="
sha256sum /work/out/h3-image-bench/2048-1.latent.f32 /work/out/h3-image-bench-base/2048-1.latent.f32
cmp /work/out/h3-image-bench/2048-1.latent.f32 /work/out/h3-image-bench-base/2048-1.latent.f32 && echo "latent byte-equal: PASS" || { echo "latent byte-equal: FAIL"; status=1; }
cat /work/out/h3-image-bench/SUMMARY.txt 2>/dev/null || :
exit $status
