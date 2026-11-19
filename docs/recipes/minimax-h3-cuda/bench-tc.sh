#!/bin/sh
# bench-script.sh with the decoder's products and attention on tensor cores (--tc): its own TOP job,
# beside the default run's, for the same-latent PSNR and warm-wall comparison.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/bench-tc.sh"
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
H3_DECODER=tc sh "$here/bench-script.sh"
# the episode-1 Mac picture and latent beside this head's --tc 2048², for the picture gates' compare
# (tools/zig/h3_image_compare.py on the fetched /work/out)
cp /assets/drama3-pilot/ep1-mac-fast/2048-warm1.png /work/out/h3-image-bench/2048-mac.png
cp /assets/drama3-pilot/h3-image-runs/34/t2i-2048.latent.f32 /work/out/h3-image-bench/2048-mac.latent.f32
