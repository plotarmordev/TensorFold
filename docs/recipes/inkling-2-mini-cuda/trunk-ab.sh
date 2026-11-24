#!/bin/sh
# Card 251's trunk lever on TOP's GB10: the same build serves Mini with bf16 attention projections and unembedding,
# then with them rounded to MXFP8 at load (TF_INKLING2_TRUNK=mxfp8), at 1 and 4 streams; then MXFP8's 32 greedy
# tokens against the float64 bf16 reference.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/trunk-ab.sh
# Staging as run-script.sh: the checkpoint in the pilot folder, the toolchain archives beside it.
set -eu
pilot=/assets/drama3-pilot
staged=$pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
out=/work/out/inkling2-trunk
model=
for dir in "$pilot/Inkling-2-Mini" "$pilot/models/Inkling-2-Mini"; do
    [ -f "$dir/config.json" ] && model=$dir && break
done
[ -n "$model" ] || { echo "no Inkling-2-Mini/config.json in $pilot" >&2; exit 1; }
if [ -z "${INKLING2_TIMED:-}" ]; then
    (cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
    tar -C /work -xJf "$staged/$zig.tar.xz"
    tar -C /work -xJf "$staged/$gcc.tar.xz"
    export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
    export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
    zig build install native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/sm121
    INKLING2_TIMED=1 exec timeout 600 sh "$0" "$@"
fi
rm -rf "$out"
mkdir -p "$out"
for trunk in bf16 mxfp8; do
    TF_INKLING2_TRUNK=$trunk /work/sm121/native/bin/tensorfold-native serve "$model" --backend cuda --host 127.0.0.1 \
        --port 8188 --name inkling-2-mini --parallel 4 --context 8192 --no-update-check >"$out/server-$trunk.log" 2>&1 &
    server=$!
    trap 'kill $server 2>/dev/null || :' EXIT
    until python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8188/v1/models", timeout=5)' 2>/dev/null; do
        kill -0 $server || { cat "$out/server-$trunk.log"; exit 1; }
        sleep 1
    done
    for level in 1 4; do
        python3 tools/bench_concurrent.py http://127.0.0.1:8188 inkling-2-mini --levels "$level" --tokens 256 --reps 3 \
            --temperatures 0 --label "$trunk-$level" --output "$out/$trunk-$level.json"
        python3 -c 'import json, sys
for c in json.load(open(sys.argv[1]))["cells"]:
    assert not c.get("failed"), c
    print("RESULT trunk", sys.argv[2], sys.argv[3], "stream(s)", c["prompt"], "decode", c["per_stream_tps"], "tok/s a stream,", c["aggregate_tps"], "total")' \
            "$out/$trunk-$level.json" "$trunk" "$level"
    done
    kill $server
    wait $server || :
done
TF_INKLING2_TRUNK=mxfp8 /work/sm121/bin/tf-cuda-test inkling2-mini-ref "$model" 32 \
    "The three laws of thermodynamics, explained for a curious student:" || echo "RESULT trunk mxfp8: tokens differ from the bf16 reference"
