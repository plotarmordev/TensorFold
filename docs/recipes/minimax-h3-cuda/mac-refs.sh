#!/bin/sh
# Where episode 1's Mac reference picture and latent sit in the pilot folder: the listing lands in
# /work/out/mac-refs/ for `job fetch`, and the first hit of each wanted name beside it.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/mac-refs.sh"
set -eu
out=/work/out/mac-refs
mkdir -p "$out"
find /assets/drama3-pilot \( -iname "*.png" -o -iname "*.latent*" \) > "$out/files.log"
for want in 2048-warm1.png t2i-2048.latent.f32; do
    hit=$(find /assets/drama3-pilot -name "$want" | head -1)
    [ -n "$hit" ] && cp "$hit" "$out/$want"
done
