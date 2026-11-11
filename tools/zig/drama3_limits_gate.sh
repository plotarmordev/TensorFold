#!/bin/sh
# drama3_limits_gate.sh STATUS LOG: every launch and control is within limits and d3_router_compact was audited on all
# five SMs. Stale records fail no launch: d3_router_compact's is pending, others belong to their own launches' receipts.
status=$1 log=$2
grep -Eq '^(FAIL|MISSED) limits' "$log" && exit 1
[ "$(grep -c '^PASS limits sm_[0-9]* drama3 d3_router_compact: ' "$log")" -eq 5 ] || exit 1
[ "$status" -eq 0 ] && exit 0
grep -qx "FAIL kernel-contract-limits: $(grep -c '^PASS limits .*(record stale)$' "$log") problems" "$log"
