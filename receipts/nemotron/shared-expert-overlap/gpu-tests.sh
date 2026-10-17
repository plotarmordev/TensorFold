#!/bin/sh
# tf-cuda-test shared-fork on the tip, the base with the test alone and the tip without the join; window-profile on base and tip.
set -u
cd "$(dirname "$0")"
M=$HOME/models/nemotron-3.5-super-vl-nvfp4
IDS=$PWD/long_document.ids.txt
export TF_HEAT_HIGH=85 TF_HEAT_LOW=75
mkdir -p gpu
banner() { echo "source $1 ($2), tf-cuda-test sha256 $(sha256sum $3/tf-cuda-test | cut -d' ' -f1)"; }
for item in tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481:tip testonly:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba:"base + the test alone" mut:14c78e735cc9effb2dfb2c2dc53c54b93d06f481:"tip without the join"; do
    b=${item%%:*} rest=${item#*:} commit=${rest%%:*} what=${rest#*:}
    { banner $commit "$what" $b; flock -w 7200 <gpu.lock> timeout 20m ./$b/tf-cuda-test shared-fork $M 2>&1; echo "exit=$?"; } > gpu/shared-fork.$b.log
done
for item in base:ed44eefbaefdfc6b52c0740ac9aeda3bbc5a36ba tip:14c78e735cc9effb2dfb2c2dc53c54b93d06f481; do
    b=${item%%:*} commit=${item#*:}
    { banner $commit $b $b; flock -w 7200 <gpu.lock> timeout 20m ./$b/tf-cuda-test window-profile $M 1,4,16 $IDS 2>&1; echo "exit=$?"; } > gpu/window-profile.$b.log
done
nvidia-smi --query-compute-apps=pid --format=csv,noheader
