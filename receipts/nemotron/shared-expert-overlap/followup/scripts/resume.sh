#!/bin/sh
# Resumed against fresh on the tip, default prompt cache: a second chat turn resumed from the first, then on a fresh server.
set -u
cd "$(dirname "$0")"
for item in turns-1:turns fresh-1:fresh; do
    label=${item%%:*} mode=${item#*:}
    flock -w 7200 <gpu.lock> timeout 20m python3 hold.py $label $PWD/../builds/tip/tensorfold-native 14c78e735cc9effb2dfb2c2dc53c54b93d06f481 $PWD/../builds/tools $HOME $mode
    echo "$label exit=$?"
done
