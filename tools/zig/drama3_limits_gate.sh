#!/bin/sh
# drama3_limits_gate.sh STATUS LOG: a clean audit passes, or one whose only problem is d3_router_compact's record.
status=$1 log=$2
[ "$status" -eq 0 ] && exit 0
grep -Eq '^(FAIL|MISSED) limits' "$log" && exit 1
stale=$(grep -c '(record stale)' "$log")
router=$(grep -c '^PASS limits sm_[0-9]* drama3 d3_router_compact: .*(record stale)$' "$log")
[ "$stale" -gt 0 ] && [ "$stale" -eq "$router" ] && grep -qx "FAIL kernel-contract-limits: $router problems" "$log"
