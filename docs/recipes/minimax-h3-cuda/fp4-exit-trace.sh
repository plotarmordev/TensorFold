#!/bin/sh
# Card 147 diagnostic: the --fp4 server's exit after one picture, under gdb for the backtrace of its 139.
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=/work/out/h3-147-trace
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
rm -rf "$out"
mkdir -p "$out"
echo "$pilot/prompt.txt $out/fox.safetensors 2048x2048 42" | "$bin/tf-h3-image-prompt-cuda" "$model/text_encoder" "$model" --serve >/dev/null 2>"$out/prompt.log"
command -v gdb || echo "no gdb"
for flag in --fp4 ""; do
    echo "=== server $flag"
    echo "$out/fox.safetensors $out/fox$flag" >"$out/req"
    if command -v gdb >/dev/null; then
        gdb -q -batch -ex run -ex bt -ex "info threads" --args "$bin/tf-h3-image-cuda" "$model" --serve $flag <"$out/req" 2>&1 | grep -vE "^\[(New|Thread)|^step |^  " | tail -60
    else
        status=0
        "$bin/tf-h3-image-cuda" "$model" --serve $flag <"$out/req" 2>"$out/server$flag.log" || status=$?
        tail -5 "$out/server$flag.log"
        echo "exit $status"
    fi
done
