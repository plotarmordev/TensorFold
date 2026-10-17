#!/bin/sh
# holds.sh MODE:LABEL:DIR:FRESH ...: each a fresh server in its own GPU hold (queued on the lock), receipts in node/receipts.
set -u
here=<work>
for item in "$@"; do
    mode=${item%%:*} rest=${item#*:} label=${rest%%:*} rest=${rest#*:} dir=${rest%%:*} fresh=${rest#*:}
    flock -w 7200 <gpu.lock> timeout 20m python3 "$here/node/hold.py" "$mode" "$label" "$here/$dir/tensorfold-native" "$(head -1 "$here/$dir/commit")" "$here/tools" "$fresh"
    echo "$label exit=$?"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader
done
