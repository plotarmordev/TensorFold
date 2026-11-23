#!/bin/sh
# Card 204's check on TOP's Spark, one job (job clones carry no history to check main out from): the TMA int8
# products' contract gates 1-4 (h3_gemm_int8_tma at its 8- and 72-column edge tiles, 16x392x128 and 300x1352x256,
# and h3_swiglu_int8_tma) on main's product (bench-204-base.patch reverse-applied), which must go red, then on this
# commit, which must pass; then the picture gate: bench-script.sh on this commit and on main's product, the pilot
# prompt and one unseen prompt at 2048², H3_PROFILE off. The served projections' N are whole 128-column tiles, so
# every latent and PNG must be main's bytes; gate.py's cosine and PSNR against main are quoted too.
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/check-204.sh
set -eu
here=docs/recipes/minimax-h3-cuda
patch_file=$here/bench-204-base.patch
out=/work/out
prompts="/assets/drama3-pilot/h3-image-pilot/prompt.txt $here/prompts/harbour.txt"
status=0
contract() { # contract SIDE: the int8_tma gates into contract-204-SIDE.log, their lines shown; returns its exit
    s=0
    H3_BUILD=contract H3_CONTRACT=int8_tma sh $here/cuda-build.sh >"$out/contract-204-$1.log" 2>&1 || s=$?
    grep -E "^(PASS|FAIL|N/A)|SUMMARY|panic|error:" "$out/contract-204-$1.log" | grep -v "limits"
    echo "contract $1 exit $s"
    return "$s"
}
bench() { # bench SIDE: the two runs into /work/out/h3-204-SIDE
    H3_BENCH_OUT=$out/h3-204-$1 H3_PROFILE= H3_BENCH_SKIP_COMPARE=1 H3_BENCH_PROMPTS="$prompts" \
        sh $here/bench-script.sh || status=$?
}
git -c safe.directory='*' apply -R $patch_file
echo "=== main's product (patch -R): must be red ==="
if contract main; then echo "main's product passed: no proof"; status=1; fi
echo "=== main's product: pictures ==="
bench main
git -c safe.directory='*' apply $patch_file
echo "=== this commit $(git -c safe.directory='*' rev-parse --short HEAD) ==="
contract card || status=1
bench card
echo "=== pictures, card against main ==="
for run in 1 2; do
    c=$out/h3-204-card/2048-$run
    m=$out/h3-204-main/2048-$run
    python3 $here/gate.py "$m" "$c" | tee $out/h3-204-gate-$run.txt
    grep -q ": PASS$" $out/h3-204-gate-$run.txt || status=1
    for f in latent.f32 png; do
        cmp -s "$c.$f" "$m.$f" && echo "2048-$run.$f identical" || { echo "2048-$run.$f DIFFERS"; status=1; }
    done
done
echo "check-204 exit $status"
exit "$status"
