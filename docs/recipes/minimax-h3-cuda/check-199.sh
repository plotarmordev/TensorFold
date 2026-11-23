#!/bin/sh
# Card 199's kernel check on TOP's Spark: each H3 kernel suite's contract gates 1-4 and live gate 5 on sm_121 in
# its own tf-cuda-test process (an abort in one leaves the others' results), then the five-SM limits audit of the
# H3 images; failures, SUMMARY lines and the audit's problem lines (stale records included) past the Desk's 200 log lines.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-199.sh
set -eu
out=/work/out
status=0
for suite in h3_ h3d_ h3p_ h3m_; do
    s=0
    # the first suite's job builds tf-cuda-test (cuda-build.sh's contract mode); the others reuse it
    if [ -x /work/contract/bin/tf-cuda-test ]; then
        /work/contract/bin/tf-cuda-test kernel-contract "$suite" >"$out/contract-$suite.log" 2>&1 || s=$?
    else
        H3_BUILD=contract H3_CONTRACT=$suite sh docs/recipes/minimax-h3-cuda/cuda-build.sh >"$out/contract-$suite.log" 2>&1 || s=$?
    fi
    grep -E "^FAIL|SUMMARY|panic|corrupted|double free|Segmentation|Aborted|error:" "$out/contract-$suite.log" | sort -u
    echo "suite $suite exit $s"
    [ "$s" = 0 ] || status=1
done
s=0
H3_BUILD=limits sh docs/recipes/minimax-h3-cuda/cuda-build.sh >"$out/audit.log" 2>&1 || s=$?
grep -E "^FAIL|record stale|^MISSED" "$out/limits.log" || tail -n 20 "$out/audit.log"
tail -n 1 "$out/limits.log"
echo "limits audit exit $s"
[ "$s" = 0 ] || status=1
echo "check-199 exit $status"
exit "$status"
