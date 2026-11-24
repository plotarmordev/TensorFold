#!/bin/sh
# Mainline llama.cpp b12651 against Unsloth's b12651-mix-c7059d6 on TOP's GB10, same flags, same GGUF (card 232's
# UD-Q4_K_M), 1 and 4 streams, for card 251's study:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-llamacpp/job.sh
# stage.sh put both source trees in the pilot folder inkling-llamacpp-251/. Both build here with card 232's flags and
# run on the staged toolchain's glibc, as 232's llama-cpp-build.sh explains. Only the measurement runs under
# `timeout 600`.
set -eu
pilot=/assets/drama3-pilot
src=$pilot/inkling-llamacpp-251
gguf=$pilot/inkling-gguf/Inkling-2-Mini-UD-Q4_K_M.gguf
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
cmake=cmake-3.31.6-linux-aarch64
out=/work/out/inkling2-llamacpp
[ -d "$src" ] || { echo "missing $src: the Desk runs stage.sh; pilot folder holds:" >&2; ls "$pilot" >&2; exit 1; }
sha256sum -c --quiet <<EOF
78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae  $gguf
b4cc788d63112b2749b40627e719eb5d3b8ed8f00c36d77189f4019cfe64bc9e  $pilot/inkling-llamacpp/$cmake.tar.gz
f98dbe546ef7633d38400648244b981e9ef36edf4ac140c3c9721fb37f483702  $src/mainline-b12651.tar.gz
dedb1c001a6d4af567d9ea152f7fdd0a467cc04486021aa0f126d7dbe2375bb0  $src/unsloth-b12651-mix-c7059d6.tar.gz
EOF
tar -C /work -xJf "$pilot/cuda-build-toolchain/$gcc.tar.xz"
tar -C /work -xzf "$pilot/inkling-llamacpp/$cmake.tar.gz"
cc=/work/$gcc/bin/aarch64-none-linux-gnu
cuda_lib=/usr/local/cuda/targets/sbsa-linux/lib
rt=/work/$gcc/aarch64-none-linux-gnu/libc
rm -rf "$out"
mkdir -p "$out"
# build NAME: unpacks NAME's tarball and leaves its llama-server in /work/NAME/build/bin
build() {
    mkdir -p "/work/$1"
    tar -C "/work/$1" --strip-components=1 -xzf "$src/$1.tar.gz"
    start=$(date +%s)
    /work/$cmake/bin/cmake -S "/work/$1" -B "/work/$1/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$cc-gcc" -DCMAKE_CXX_COMPILER="$cc-g++" \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER="$cc-g++" -DLLAMA_OPENSSL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$cuda_lib:$cuda_lib/stubs:/usr/lib/aarch64-linux-gnu" \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
    /work/$cmake/bin/cmake --build "/work/$1/build" -j "$(nproc)" --target llama-server >/dev/null
    echo "build $1: $(($(date +%s) - start)) s"
}
build mainline-b12651
build unsloth-b12651-mix-c7059d6
export gguf out rt cuda_lib
start=$(date +%s)
timeout 600 sh -eu -c '
    for name in mainline-b12651 unsloth-b12651-mix-c7059d6; do
        bin=/work/$name/build/bin
        "$rt/lib/ld-linux-aarch64.so.1" --library-path "$rt/lib64:$rt/usr/lib64:$bin:$cuda_lib:/usr/lib/aarch64-linux-gnu" \
            "$bin/llama-server" -m "$gguf" --jinja -fa on -ngl 99 -c 32768 --parallel 4 --host 127.0.0.1 --port 8189 \
            --no-webui >"$out/$name.log" 2>&1 &
        server=$!
        until python3 -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8189/health\", timeout=5)" 2>/dev/null; do
            kill -0 "$server" 2>/dev/null || { tail -n 40 "$out/$name.log"; exit 1; }
            sleep 1
        done
        for level in 1 4; do
            python3 tools/bench_concurrent.py http://127.0.0.1:8189 inkling-2-mini --levels "$level" --tokens 256 \
                --reps 3 --temperatures 0 --label "$name-$level" --output "$out/$name-$level.json" >/dev/null
            python3 -c "import json, sys; cells = json.load(open(sys.argv[1]))[\"cells\"]
assert not any(c.get(\"failed\") for c in cells), \"a request failed\"
for c in cells: print(\"RESULT {} gb10 {} stream(s) {}: decode {} tok/s a stream, {} total\".format(sys.argv[2], sys.argv[3], c[\"prompt\"], c[\"per_stream_tps\"], c[\"aggregate_tps\"]))" \
                "$out/$name-$level.json" "$name" "$level"
        done
        kill "$server"
        wait "$server" || :
    done
'
echo "measure: $(($(date +%s) - start)) s"
