#!/bin/sh
# Inkling 2 Mini on TOP's GB10, llama.cpp's side of the README's comparison:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/compare.sh
# Prints the bytes of the weights each engine loads, builds llama-server, then measures it under peak-memory.py with
# the measured part capped at 10 minutes. TensorFold's side is run-script.sh's own job (its float64 reference alone
# takes over 4 minutes); the README sets the two side by side.
set -eu
here=docs/recipes/inkling-2-mini-cuda
pilot=/assets/drama3-pilot
bytes() { stat -L -c %s "$@" 2>/dev/null | awk '{s += $1} END {printf "%.0f bytes (%.3f GiB)", s, s / 2^30}'; }
echo "RESULT tensorfold bf16 checkpoint: $(bytes "$pilot"/Inkling-2-Mini/*.safetensors "$pilot"/models/Inkling-2-Mini/*.safetensors)"
echo "RESULT llama.cpp UD-Q4_K_M GGUF: $(bytes "$pilot/inkling-gguf/Inkling-2-Mini-UD-Q4_K_M.gguf")"
start=$(date +%s)
sh "$here/llama-cpp-build.sh"
echo "build: $(($(date +%s) - start)) s"
start=$(date +%s)
# llama-server runs as the toolchain's loader (llama-cpp-build.sh), so that is the program peak-memory.py watches
timeout 600 python3 "$here/peak-memory.py" ld-linux-aarch64.so.1 sh "$here/llama-cpp-baseline.sh"
echo "measure: $(($(date +%s) - start)) s"
