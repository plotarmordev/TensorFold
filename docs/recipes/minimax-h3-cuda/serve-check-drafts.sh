#!/bin/sh
# serve-check.sh's discipline for the draft path: the pilot's prompt at 1024², served twice under
# --draft-steps 1 and twice without, then fresh renders of both. Served pairs must be byte-identical
# within a flag (the path is deterministic), served must equal fresh, and draft must differ from
# plain. ~5 min after the build; exits 1 on a miss.
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
"$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" "$pilot/prompt.txt" "$out/c.safetensors" 1024x1024 42 262144
printf '%s\n' "$out/c.safetensors $out/d1" "$out/c.safetensors $out/d2" |
    "$bin/tf-h3-image-cuda" "$model" --serve --draft-steps 1 >"$out/replies-d" 2>"$out/serve-d.log"
cat "$out/replies-d"; tail -20 "$out/serve-d.log" >&2
printf '%s\n' "$out/c.safetensors $out/p1" "$out/c.safetensors $out/p2" |
    "$bin/tf-h3-image-cuda" "$model" --serve >"$out/replies-p" 2>"$out/serve-p.log"
cat "$out/replies-p"; tail -20 "$out/serve-p.log" >&2
"$bin/tf-h3-image-cuda" "$model" "$out/c.safetensors" "$out/fresh-plain" 2>"$out/fresh-plain.log"
"$bin/tf-h3-image-cuda" "$model" "$out/c.safetensors" "$out/fresh-draft" --draft-steps 1 2>"$out/fresh-draft.log"
status=0
check() { if cmp -s "$1" "$2"; then echo "$3"; else echo "MISMATCH: $3 ($1 vs $2)"; status=1; fi }
check "$out/d1.latent.f32" "$out/d2.latent.f32" "served drafts identical"
check "$out/p1.latent.f32" "$out/p2.latent.f32" "served plains identical"
check "$out/p1.latent.f32" "$out/fresh-plain.latent.f32" "served plain = fresh plain"
check "$out/d1.latent.f32" "$out/fresh-draft.latent.f32" "served draft = fresh draft"
cmp -s "$out/d1.latent.f32" "$out/p1.latent.f32" && { echo "draft = plain: drafts drew the default picture"; status=1; }
echo "serve-check-drafts exit $status"
exit "$status"
