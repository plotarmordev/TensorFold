#!/bin/sh
# kernel-contract-limits over every image of the five release SMs, as a TOP job:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/kernel-contract/limits-job.sh
# Prints each stale or failing check and the RECORD line of its launch, the values its record is replaced with.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
status=0
H3_BUILD=limits LIMITS_IMAGES= sh "$here/../recipes/minimax-h3-cuda/cuda-build.sh" || status=$?
log=/work/out/limits.log
grep -E '^(INFO|SUMMARY|MISSED|RED )' "$log"
awk '/^(PASS|FAIL) limits sm_[0-9]+ .*\(record stale\)|^FAIL limits/ { sub(/^(PASS|FAIL) limits sm_[0-9]+ /, ""); sub(/: [0-9]+ registers,.*/, ""); stale[$0] = 1; print "STALE " $0 }
    /^RECORD / { name = substr($0, 8); sub(/: \.usage = .*/, "", name); if (name in stale) print }' "$log" | sort -u
exit "$status"
