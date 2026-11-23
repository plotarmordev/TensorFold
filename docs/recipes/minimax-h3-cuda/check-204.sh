#!/bin/sh
# Card 204's contract check on TOP's Spark, one job (job clones carry no history to check main out from): the TMA int8
# products' gates 1-4 on this commit, which must pass, then on main's product (bench-204-base.patch reverse-applied),
# which must fail gate 3 at each edge tile on its own: 16x392x128 with output 384 (the first of its unwritten
# columns 384-391) NaN, then - that served line dropped, as the gate stops at a kernel's first red shape -
# 300x1352x256, and SwiGLU's 300x776x256 against main's plain product then h3_swiglu. Capped at 10 minutes.
# The picture gate (2048², card against main: latents and PNGs byte-identical) ran at add4bf341's same kernel
# through this script's earlier form (job top-a4be033f62db).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-204.sh
set -eu
[ -n "${CHECK_204_CAPPED:-}" ] || CHECK_204_CAPPED=1 exec timeout 600 sh "$0"
here=docs/recipes/minimax-h3-cuda
contract=zig/tests/cuda/contract_h3_int8.zig
out=/work/out
status=0
contract() { # contract SIDE: the int8_tma gates into contract-204-SIDE.log, their lines shown; returns its exit
    s=0
    H3_BUILD=contract H3_CONTRACT=int8_tma sh $here/cuda-build.sh >"$out/contract-204-$1.log" 2>&1 || s=$?
    grep -E "^(PASS|FAIL|N/A) (nonfinite|repeat|shapes|sums)|SUMMARY|panic|error:" "$out/contract-204-$1.log"
    echo "contract $1 exit $s"
    return "$s"
}
red() { # red SIDE PATTERN: SIDE's gate 3 must fail as PATTERN says
    grep -qE "^FAIL shapes cuda h3 int8 tma $2" "$out/contract-204-$1.log" || { echo "$1: no gate-3 failure $2"; status=1; }
}
echo "=== this commit $(git -c safe.directory='*' rev-parse --short HEAD): must pass ==="
contract card || status=1
git -c safe.directory='*' apply -R $here/bench-204-base.patch
echo "=== main's product (patch -R): red at 16x392x128 ==="
contract main || true
red main "product .*: at 16x392x128 output 384 is -?nan "
red main "swiglu .*: at 300x776x256 its bytes differ "
echo "=== main's product, 16x392x128 dropped: red at 300x1352x256 ==="
sed -i '/\.rows = 16, \.n = 392, \.k = 128/d' $contract
contract main-1352 || true
red main-1352 "product .*: at 300x1352x256 output [0-9]+ is -?nan "
git -c safe.directory='*' checkout -- $contract
echo "check-204 exit $status"
exit "$status"
