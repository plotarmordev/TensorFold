#!/bin/sh
# The warm --serve path past its second picture: two plain serves, then two serves with
# --draft-steps 3, each pair read back against fresh single-shot renders. Caught the kept-AdaLN
# crash (the second request walked the first's freed tables) and stands as its regression check.
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
status=0
# plain serves, then draft serves; each pair must reach its second reply
for flag in plain drafts; do
    extra=
    [ "$flag" = drafts ] && extra="--draft-steps 3"
    printf '%s\n' "$out/c.safetensors $out/$flag-1" "$out/c.safetensors $out/$flag-2" |
        "$bin/tf-h3-image-cuda" "$model" --serve $extra >"$out/replies-$flag" 2>"$out/serve-$flag.log" || status=1
    cat "$out/replies-$flag"
    n=$(grep -c done "$out/replies-$flag" || :)
    echo "$flag: $n/2 replies"
    [ "$n" = 2 ] || status=1
done
echo "serve-check-drafts exit $status"
exit "$status"
