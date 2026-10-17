#!/bin/sh
# base, PR, PR, base (full holds), then PR, base, base, PR (cold prefill only); each a fresh server in its own GPU hold.
set -u
cd "$(dirname "$0")"
for item in qualify:base-1:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba qualify:pr-1:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481 qualify:pr-2:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481 qualify:base-2:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba cold:pr-3:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481 cold:base-3:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba cold:base-4:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba cold:pr-4:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481; do
    script=${item%%:*} rest=${item#*:} label=${rest%%:*} rest=${rest#*:} dir=${rest%%:*} commit=${rest#*:}
    flock -w 7200 <gpu.lock> timeout 20m python3 $script.py $label $PWD/$dir/tensorfold-native $commit $PWD/tools
    echo "$label exit=$?"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader
done
