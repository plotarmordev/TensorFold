#!/bin/sh
# Bisect for the warm-serve crash: p1 then p2 under H3_PROFILE (eager, no graph capture), then a
# graphed pair, then four --draft-steps 3 serves and fresh renders. Served runs must be
# byte-identical to each other and to the fresh render of the same flag. ~6 min after the build;
# exits 1 on a miss.
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/serve-check-drafts.sh"
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=/work/out/h3-serve-check-drafts
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
rm -rf "$out"
mkdir -p "$out"
"$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$out/c.safetensors" 2048x2048 42 262144
printf '%s\n' "$out/c.safetensors $out/e1" "$out/c.safetensors $out/e2" |
    H3_PROFILE=1 "$bin/tf-h3-image-cuda" "$model" --serve >"$out/replies-e" 2>"$out/serve-e.log"
cat "$out/replies-e"; tail -20 "$out/serve-e.log" >&2
printf '%s\n' "$out/c.safetensors $out/p1" "$out/c.safetensors $out/p2" |
    "$bin/tf-h3-image-cuda" "$model" --serve >"$out/replies-p" 2>"$out/serve-p.log"
cat "$out/replies-p"; tail -20 "$out/serve-p.log" >&2
printf '%s\n' "$out/c.safetensors $out/d1" "$out/c.safetensors $out/d2" |
    "$bin/tf-h3-image-cuda" "$model" --serve --draft-steps 3 >"$out/replies-d" 2>"$out/serve-d.log"
cat "$out/replies-d"; tail -40 "$out/serve-d.log" >&2
"$bin/tf-h3-image-cuda" "$model" "$out/c.safetensors" "$out/fresh-plain" 2>"$out/fresh-plain.log"
status=0
check() { if cmp -s "$1" "$2"; then echo "$3"; else echo "MISMATCH: $3 ($1 vs $2)"; status=1; fi }
check "$out/e1.latent.f32" "$out/e2.latent.f32" "eager serves identical"
check "$out/e1.latent.f32" "$out/p1.latent.f32" "eager = graphed"
check "$out/p1.latent.f32" "$out/p2.latent.f32" "served plains identical"
check "$out/p1.latent.f32" "$out/fresh-plain.latent.f32" "served plain = fresh plain"
check "$out/d1.latent.f32" "$out/d2.latent.f32" "served drafts identical"
echo "serve-check-drafts exit $status"
exit "$status"
