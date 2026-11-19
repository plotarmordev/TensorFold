#!/bin/sh
# Card 120's A/B pair in one job: the baseline is main's own tree, reproduced by copying the four
# card120-main/ files over the branch's (the door's clone is a detached single-commit checkout — no
# origin/main, no history — and a reverse git apply proved unreliable there). The patch that turns
# that baseline back into the branch is card120-main.patch; each suite's /work/out/h3-image-bench is
# kept under card120/{main,heads-fused}.
set -eu
out=/work/out/h3-image-bench
keep=/work/out/card120
patch=docs/recipes/minimax-h3-cuda/card120-main.patch
main=docs/recipes/minimax-h3-cuda/card120-main
rm -rf "$keep"
mkdir -p "$keep"

cp "$main/h3.cu" zig/kernels/cuda/h3.cu
cp "$main/h3_attention.cuh" zig/kernels/cuda/h3_attention.cuh
cp "$main/cuda.zig" zig/src/families/h3/cuda.zig
cp "$main/registry_media.zig" zig/src/core/contract/registry_media.zig
trap 'git -c safe.directory="*" apply "$patch" 2>/dev/null || :; rm -rf "$out" || :' EXIT
sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/main"

git -c safe.directory='*' apply "$patch"
sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/heads-fused"
