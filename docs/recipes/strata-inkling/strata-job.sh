#!/bin/sh
# Strata's own Inkling 2 Mini engine on TOP's GB10, for card 209's measured gap:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/strata-inkling/strata-job.sh
# The job has no network: the Desk staged strata-inkling 0.1.66 (linux-aarch64, CUDA 13; its cuBLAS, cudart and
# libgomp ride along) and the SQ-10 pack in the pilot folder strata-inkling-pack/. The engine runs as shipped, nothing is
# installed: its built-in `--bench` (DSpark drafting as the engine defaults it) at 1, 4 and 64 streams, 256 generated
# tokens each, then `--selfcheck`. Logs land in /work/out/strata-inkling; the `[bench]` lines go to stdout.
set -eu
dir=/assets/drama3-pilot/strata-inkling-pack
pack=$dir/Inkling-2-Mini-SQ-10.strata
engine=$(find "$dir" -name strata-inkling -type f | head -1)
out=/work/out/strata-inkling
[ -f "$pack" ] && [ -n "$engine" ] || { echo "missing pack or engine under $dir" >&2; exit 1; }
(cd "$dir" && grep " Inkling-2-Mini-SQ-10.strata\$" PACK-SHA256SUMS.txt | sha256sum -c --quiet)
mkdir -p "$out"
export LD_LIBRARY_PATH=$(dirname "$engine")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
"$engine" --version
status=0
for streams in 1 4 64; do
    "$engine" --bench --native "$pack" --parallel "$streams" --gen 256 \
        >"$out/bench-$streams.log" 2>&1 || status=$?
    grep '^\[bench\]' "$out/bench-$streams.log" || { tail -20 "$out/bench-$streams.log"; status=1; }
done
"$engine" --selfcheck --native "$pack" >"$out/selfcheck.log" 2>&1 || status=$?
tail -5 "$out/selfcheck.log"
exit "$status"
