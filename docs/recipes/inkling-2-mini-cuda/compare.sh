#!/bin/sh
# Inkling 2 Mini on TOP's GB10, TensorFold beside llama.cpp, in one job so both meet the same machine:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/compare.sh
# Prints the bytes of the weights each engine loads, then runs run-script.sh and llama-cpp-baseline.sh one after the
# other, each under peak-memory.py watching its server.
set -eu
here=docs/recipes/inkling-2-mini-cuda
pilot=/assets/drama3-pilot
bytes() { stat -L -c %s "$@" 2>/dev/null | awk '{s += $1} END {printf "%d bytes (%.3f GiB)", s, s / 2^30}'; }
echo "RESULT tensorfold bf16 checkpoint: $(bytes "$pilot"/Inkling-2-Mini/*.safetensors "$pilot"/models/Inkling-2-Mini/*.safetensors)"
echo "RESULT llama.cpp UD-Q4_K_M GGUF: $(bytes "$pilot/inkling-gguf/Inkling-2-Mini-UD-Q4_K_M.gguf")"
python3 "$here/peak-memory.py" tensorfold-native sh "$here/run-script.sh"
python3 "$here/peak-memory.py" llama-server sh "$here/llama-cpp-baseline.sh"
