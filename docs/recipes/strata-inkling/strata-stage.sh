#!/bin/sh
# Stages Strata's own Inkling 2 Mini engine and pack for strata-job.sh, whose GPU job has no network: the
# strata-inkling 0.1.66 linux-aarch64 CUDA 13 release (Strata v0.2.20, MIT, https://github.com/Niko1221/Strata) and
# the SQ-10 pack of strata-engine/Inkling-2-Mini-Strata (10.47 GB, the pack of Strata's 74.3 tok/s GB10 solo row),
# pinned to the recorded revision and sha256, land in TOP's pilot folder as strata-inkling-pack/. Nothing is
# installed on TOP. The Desk runs it from any machine with network and the top-menu key, never on a Spark:
#   sh docs/recipes/strata-inkling/strata-stage.sh
set -eu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
dir=$work/strata-inkling-pack
mkdir "$dir"
release=https://github.com/Niko1221/Strata/releases/download/v0.2.20
packs=https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/04996a1e2c45009b6a04f2af126cb6bdb2ba3587
engine=strata-inkling-0.1.66-linux-aarch64-cuda-cuda13.tar.gz
pack=Inkling-2-Mini-SQ-10.strata
curl -sSfL -o "$work/$engine" "$release/$engine"
echo "a01f0e1f2e629d53ff17f1defd7164076844f62ed73cf747298afe68e43a77c1  $work/$engine" | sha256sum -c --quiet
tar -C "$dir" -xzf "$work/$engine"
curl -sSfL -o "$dir/$pack" "$packs/$pack"
echo "ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934  $dir/$pack" | sha256sum -c --quiet
tar -C "$work" -c strata-inkling-pack | ssh top-menu "put top strata-inkling-pack"
