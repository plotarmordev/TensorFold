#!/bin/sh
# Stages Strata's own Inkling 2 Mini engine and pack for strata-job.sh, whose GPU job has no network: the
# strata-inkling 0.1.66 linux-aarch64 CUDA 13 release (Strata v0.2.20, MIT, https://github.com/Niko1221/Strata) and
# the SQ-10 pack of strata-engine/Inkling-2-Mini-Strata (10.47 GB, the pack of Strata's 74.3 tok/s GB10 solo row),
# each checked against its publisher's sha256, land in TOP's pilot folder as strata-inkling-0.1.66/. Nothing is
# installed on TOP. The Desk runs it from any machine with network and the top-menu key, never on a Spark:
#   sh docs/recipes/strata-inkling/strata-stage.sh
set -eu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
dir=$work/strata-inkling-0.1.66
mkdir "$dir"
release=https://github.com/Niko1221/Strata/releases/download/v0.2.20
packs=https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/main
engine=strata-inkling-0.1.66-linux-aarch64-cuda-cuda13.tar.gz
pack=Inkling-2-Mini-SQ-10.strata
curl -sSfL -o "$work/SHA256SUMS.txt" "$release/SHA256SUMS.txt"
curl -sSfL -o "$work/$engine" "$release/$engine"
(cd "$work" && grep " $engine\$" SHA256SUMS.txt | sha256sum -c --quiet)
tar -C "$dir" -xzf "$work/$engine"
curl -sSfL -o "$dir/PACK-SHA256SUMS.txt" "$packs/SHA256SUMS.txt"
curl -sSfL -o "$dir/$pack" "$packs/$pack"
(cd "$dir" && grep " $pack\$" PACK-SHA256SUMS.txt | sha256sum -c --quiet)
tar -C "$work" -c strata-inkling-0.1.66 | ssh top-menu "put top strata-inkling-0.1.66"
