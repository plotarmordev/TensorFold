#!/bin/sh
# Each build's first start in an empty HOME of its own: the window-cost file names it writes give its cache key.
set -u
cd "$(dirname "$0")"
for item in cal-base:base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba cal-tip:tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481; do
    label=${item%%:*} rest=${item#*:} dir=${rest%%:*} commit=${rest#*:}
    rm -rf homes/$label && mkdir -p homes/$label
    flock -w 7200 <gpu.lock> timeout 20m python3 hold.py $label $PWD/../builds/$dir/tensorfold-native $commit $PWD/../builds/tools $PWD/homes/$label start
    echo "$label exit=$?"
done
