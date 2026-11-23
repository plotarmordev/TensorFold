#!/bin/sh
# Inkling 2 Mini's CUDA limits audit on TOP, compile only (no kernel runs): the five SMs' inkling2 images through
# kernel-contract-limits. Its RECORD lines, echoed into the job log, are the registry's usage for the SMs no
# earlier receipt covered (zig/src/core/contract/inkling2_experts_records.zig).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/inkling-2-mini-cuda/limits-job.sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
status=0
H3_BUILD=limits LIMITS_IMAGES="inkling2_modelopt inkling2_loop_math" sh "$here/../minimax-h3-cuda/cuda-build.sh" ||
    status=$?
grep -E '^(RECORD|FAIL|PASS) ' /work/out/limits.log || true
exit "$status"
