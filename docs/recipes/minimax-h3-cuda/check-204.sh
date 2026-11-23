#!/bin/sh
# Card 204's kernel check on TOP's Spark: the TMA int8 products' contract gates 1-4 (h3_gemm_int8_tma at its
# 8- and 72-column edge tiles, 16x392x128 and 300x1352x256, and h3_swiglu_int8_tma), then the five-SM limits
# audit's lines for both kernels, so a changed register count shows as a stale record.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-204.sh
set -eu
here=docs/recipes/minimax-h3-cuda
out=/work/out
status=0
s=0
H3_BUILD=contract H3_CONTRACT=int8_tma sh $here/cuda-build.sh >"$out/contract-204.log" 2>&1 || s=$?
grep -E "^(PASS|FAIL|N/A)|SUMMARY|panic|error:" "$out/contract-204.log" | grep -v "^PASS limits"
echo "contract exit $s"
[ "$s" = 0 ] || status=1
s=0
H3_BUILD=limits sh $here/cuda-build.sh >"$out/audit-204.log" 2>&1 || s=$?
grep -E "limits .*int8_tma|^RECORD h3 h3_(gemm|swiglu)_int8_tma" "$out/limits.log"
echo "limits audit exit $s"
[ "$s" = 0 ] || status=1
echo "check-204 exit $status"
exit "$status"
