#!/bin/sh
# One resident --serve transformer against fresh runs: its AdaLN tables kept across pictures with the same timesteps
# and replaced when a case's differ must leave every latent byte-identical to the case rendered alone.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/serve-check.sh"
# Case a: the pilot's prompt at 512²; case b: a with a 2 x 2 reference grid, whose held rows add the keyframe
# timestep to the table. Served order a a b a: a build, a reuse, a replacement and a return. ~5 min; exits 1 on a miss.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=/work/out/h3-serve-check
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
rm -rf "$out"
mkdir -p "$out"
"$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$out/a.safetensors" 512x512 42 262144
python3 - "$out/a.safetensors" "$out/b.safetensors" <<'EOF'
import sys
import numpy as np
from safetensors.numpy import load_file, save_file
case = load_file(sys.argv[1])
case["references"] = np.array([[2, 2]], dtype=np.int32)
case["anchors"] = np.random.default_rng(7).standard_normal((1, 96)).astype(np.float32)
save_file(case, sys.argv[2])
EOF
for c in a b; do "$bin/tf-h3-image-cuda" "$model" "$out/$c.safetensors" "$out/fresh-$c"; done
printf '%s\n' "$out/a.safetensors $out/served-1" "$out/a.safetensors $out/served-2" \
    "$out/b.safetensors $out/served-3" "$out/a.safetensors $out/served-4" |
    "$bin/tf-h3-image-cuda" "$model" --serve >"$out/replies"
cat "$out/replies"
status=0
for pair in 1:a 2:a 3:b 4:a; do
    if cmp "$out/served-${pair%:*}.latent.f32" "$out/fresh-${pair#*:}.latent.f32"; then echo "served ${pair%:*} = fresh ${pair#*:}"; else status=1; fi
done
cmp -s "$out/fresh-a.latent.f32" "$out/fresh-b.latent.f32" && { echo "a and b render alike: b proves no replacement"; status=1; }
echo "serve-check exit $status"
exit "$status"
