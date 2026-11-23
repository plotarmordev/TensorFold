#!/bin/sh
# kernel-contract-limits over every image of the five release SMs, as a TOP job:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/kernel-contract/limits-job.sh
# nvcc builds the images with -Xptxas -v, the build stopping at 600 s. Prints each stale or failing check and the
# RECORD line of its launch (the values its record is replaced with), the router's measured lines, then every entry
# function's ptxas resources as `PTXAS sm_N symbol: Used ...`.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
status=0
NVCC_APPEND_FLAGS="-Xptxas -v" H3_BUILD=limits LIMITS_IMAGES= timeout 600 \
    sh "$here/../recipes/minimax-h3-cuda/cuda-build.sh" >/work/out/build.log 2>&1 || status=$?
[ "$status" = 124 ] && echo "limits job stopped at its 600 s limit"
tail -n 4 /work/out/build.log
log=/work/out/limits.log
grep -E '^(INFO|SUMMARY|MISSED|RED )' "$log"
awk '/^(PASS|FAIL) limits sm_[0-9]+ .*\(record stale\)|^FAIL limits/ { sub(/^(PASS|FAIL) limits sm_[0-9]+ /, ""); sub(/: [0-9]+ registers,.*/, ""); stale[$0] = 1; print "STALE " $0 }
    /^RECORD / { name = substr($0, 8); sub(/: \.usage = .*/, "", name); if (name in stale) print }' "$log" | sort -u
grep -E '^(PASS|FAIL) limits sm_[0-9]+ router tf_nemo_router_tiles:' "$log"
awk -F"'" '/Compiling entry function/ { fn = $2; sm = $4; next }
    fn != "" && /ptxas info *: Used / { sub(/.*Used /, "Used "); print "PTXAS " sm " " fn ": " $0; fn = "" }' /work/out/build.log | sort -u
exit "$status"
