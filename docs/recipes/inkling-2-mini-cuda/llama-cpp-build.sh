#!/bin/sh
# Builds llama.cpp b12611's CUDA llama-server into /work/llama-build for llama-cpp-baseline.sh. The job has no network:
# the Desk staged Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file (renamed .gguf) in the pilot folder's
# inkling-gguf, and llama.cpp b12611 (690b87d88, ggml-org #32704's merge) with CMake 3.31.6's linux-aarch64 archive in
# inkling-llamacpp.
set -eu
pilot=/assets/drama3-pilot
llama=$pilot/inkling-llamacpp
staged=$pilot/cuda-build-toolchain
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
cmake=cmake-3.31.6-linux-aarch64
sha256sum -c --quiet <<EOF
78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae  $pilot/inkling-gguf/Inkling-2-Mini-UD-Q4_K_M.gguf
b4cc788d63112b2749b40627e719eb5d3b8ed8f00c36d77189f4019cfe64bc9e  $llama/$cmake.tar.gz
EOF
tar -C /work -xJf "$staged/$gcc.tar.xz"
tar -C /work -xzf "$llama/$cmake.tar.gz"
cc=/work/$gcc/bin/aarch64-none-linux-gnu
cuda_lib=/usr/local/cuda/targets/sbsa-linux/lib
# the staged gcc's ld searches only its own sysroot: point it at the toolkit and the host driver for libggml-cuda's needs
/work/$cmake/bin/cmake -S "$llama/llama.cpp-b12611" -B /work/llama-build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$cc-gcc" -DCMAKE_CXX_COMPILER="$cc-g++" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER="$cc-g++" -DLLAMA_OPENSSL=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$cuda_lib:$cuda_lib/stubs:/usr/lib/aarch64-linux-gnu" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
/work/$cmake/bin/cmake --build /work/llama-build -j "$(nproc)" --target llama-server >/dev/null
# loads every shared library the server needs: a loader error shows here, before the measured part
LD_LIBRARY_PATH=$cuda_lib /work/llama-build/bin/llama-server --version
