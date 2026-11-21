#!/bin/sh
# Card 147 diagnostic: the --fp4 server's exit after one picture, ReleaseSafe for the stack of its 139.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
nvcc=/usr/local/cuda/bin/nvcc
out=/work/out/h3-147-trace
zig build tf-h3-image-prompt-cuda tf-h3-image-cuda -Dsm=121 -Dnvcc=$nvcc -Doptimize=ReleaseSafe --prefix /work/safe
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
rm -rf "$out"
mkdir -p "$out"
echo "$pilot/prompt.txt $out/fox.safetensors 2048x2048 42" | /work/safe/bin/tf-h3-image-prompt-cuda "$model/text_encoder" "$model" --serve >/dev/null 2>"$out/prompt.log"
echo "$out/fox.safetensors $out/fox" >"$out/req"
status=0
/work/safe/bin/tf-h3-image-cuda "$model" --serve --fp4 <"$out/req" 2>"$out/server.log" || status=$?
grep -vE "^step |^  " "$out/server.log" | tail -60
echo "exit $status"
