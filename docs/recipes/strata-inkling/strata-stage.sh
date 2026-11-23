#!/bin/sh
# Stages Strata v0.2.19 (MIT, https://github.com/Niko1221/Strata) for strata-job.sh, whose GPU job has no network: its
# strata-inkling 0.1.65 engine and strata-control 0.1.65 server for linux-aarch64, and from
# strata-engine/Inkling-2-Mini-Strata the SQ-20 (19.35 GB) and SQ-10 (10.47 GB) packs with the tokenizer.json the
# server's tokenizer is built from. Each release file and pack is checked against its publisher's sha256; all land
# in TOP's pilot folder as strata-inkling-sq20/. Nothing is installed on TOP. The Desk runs it from any machine with
# network and the top-menu key, never on a Spark:
#   sh docs/recipes/strata-inkling/strata-stage.sh
set -eu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
dir=$work/strata-inkling-sq20
release=https://github.com/Niko1221/Strata/releases/download/v0.2.19
packs=https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/resolve/main
curl -sSfL -o "$work/SHA256SUMS.txt" "$release/SHA256SUMS.txt"
for part in inkling-0.1.65-linux-aarch64-cuda-cuda13 control-0.1.65-linux-aarch64; do
    curl -sSfL -o "$work/strata-$part.tar.gz" "$release/strata-$part.tar.gz"
    (cd "$work" && grep " strata-$part.tar.gz\$" SHA256SUMS.txt | sha256sum -c --quiet)
    mkdir -p "$dir/${part%%-*}"
    tar -C "$dir/${part%%-*}" -xzf "$work/strata-$part.tar.gz"
done
curl -sSfL -o "$dir/PACK-SHA256SUMS.txt" "$packs/SHA256SUMS.txt"
for file in Inkling-2-Mini-SQ-20.strata Inkling-2-Mini-SQ-10.strata tokenizer.json; do
    curl -sSfL -o "$dir/$file" "$packs/$file"
done
(cd "$dir" && grep -E ' Inkling-2-Mini-SQ-(20|10)\.strata$' PACK-SHA256SUMS.txt | sha256sum -c --quiet)
tar -C "$work" -c strata-inkling-sq20 | ssh top-menu "put top strata-inkling-sq20"
