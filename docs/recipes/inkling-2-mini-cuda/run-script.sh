#!/bin/sh
# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, revision ed40e7da62) whole on TOP's GB10:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/run-script.sh
# The job has no network: the checkpoint has to be in the pilot folder first, put there by the Desk:
#   tar -C MODELS -c Inkling-2-Mini | ssh top-menu "put top models/Inkling-2-Mini"
set -eu
pilot=/assets/drama3-pilot
model=$pilot/models/Inkling-2-Mini
if [ ! -f "$model/config.json" ]; then
    echo "missing $model/config.json; the pilot's models and any Inkling config:" >&2
    ls "$pilot/models" >&2 || true
    find /assets -maxdepth 5 -name config.json -exec grep -l '"inkling2' {} + >&2 || true
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
