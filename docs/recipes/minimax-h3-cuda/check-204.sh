#!/bin/sh
# Card 204's contract check on TOP's Spark: the TMA int8 products' gates 1-4, one pass a job (each is a cold
# sm_121 build, ~9 minutes; capped at 10).
# - check-204.sh: this commit, which must pass.
# - check-204-main.sh: main's product (bench-204-base.patch reverse-applied), which must fail gate 3 at
#   16x392x128 with output 384 (the first of its unwritten columns 384-391) NaN, and SwiGLU at 300x776x256 against
#   main's plain product then h3_swiglu.
# - check-204-1352.sh: main's product with 16x392x128 dropped (the gate stops at a kernel's first red shape),
#   which must fail at 300x1352x256 with output 1280 (the first of its unwritten columns 1280-1351) NaN.
# The picture gate (2048², card against main: latents and PNGs byte-identical) ran at add4bf341's same kernel
# through this script's earlier form (job top-a4be033f62db).
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-204.sh
set -eu
side=${CHECK_204_SIDE:-card}
[ -n "${CHECK_204_CAPPED:-}" ] || CHECK_204_CAPPED=1 CHECK_204_SIDE=$side exec timeout 600 sh "$0"
here=docs/recipes/minimax-h3-cuda
log=/work/out/contract-204-$side.log
red() { # red PATTERN: gate 3 must fail as PATTERN says
    grep -qE "^FAIL shapes cuda h3 int8 tma $1" "$log" || { echo "$side: no gate-3 failure $1"; status=1; }
}
[ "$side" = card ] || git -c safe.directory='*' apply -R $here/bench-204-base.patch
[ "$side" != 1352 ] || sed -i '/\.rows = 16, \.n = 392, \.k = 128/d' zig/tests/cuda/contract_h3_int8.zig
status=0
H3_BUILD=contract H3_CONTRACT=int8_tma sh $here/cuda-build.sh >"$log" 2>&1 || status=$?
grep -E "^(PASS|FAIL|N/A) (nonfinite|repeat|shapes|sums)|SUMMARY|panic|error:" "$log" || true
echo "contract $side exit $status"
case $side in
main)
    status=0
    red "product .*: at 16x392x128 output 384 is -?nan "
    red "swiglu .*: at 300x776x256 its bytes differ " ;;
1352)
    status=0
    red "product .*: at 300x1352x256 output 1280 is -?nan " ;;
esac
echo "check-204 $side exit $status"
exit "$status"
