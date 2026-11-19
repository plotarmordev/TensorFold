#!/bin/sh
# Card 119's flag-off/flag-on pair: bench-script.sh once with the bf16 attention, once with H3_ATTN=fp8's
# e4m3 attention, each suite's /work/out/h3-image-bench kept under fp8-attn/{bf16,fp8}.
set -eu
out=/work/out/h3-image-bench
keep=/work/out/fp8-attn
rm -rf "$keep"
mkdir -p "$keep"
sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/bf16"
H3_ATTN=fp8 sh docs/recipes/minimax-h3-cuda/bench-script.sh
mv "$out" "$keep/fp8"
