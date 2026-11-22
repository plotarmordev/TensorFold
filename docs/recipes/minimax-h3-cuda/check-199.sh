#!/bin/sh
# Card 199's kernel check on TOP's Spark: every H3 kernel's contract gates 1-4 and live gate 5 on sm_121
# (`kernel-contract h3`: the h3, decoder, prompt and modulation names) into /work/out/contract.log, its
# failures and SUMMARY past the Desk's 200 log lines; registers and shared bytes in the launch lines.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-199.sh
set -eu
status=0
H3_BUILD=contract H3_CONTRACT=h3 sh docs/recipes/minimax-h3-cuda/cuda-build.sh >/work/out/contract.log 2>&1 || status=$?
grep -E "FAIL|SUMMARY|panic|double free|error|Segmentation|Aborted|limits live" /work/out/contract.log | grep -v "warning" | tail -n 120
tail -n 15 /work/out/contract.log
echo "contract exit $status"
exit "$status"
