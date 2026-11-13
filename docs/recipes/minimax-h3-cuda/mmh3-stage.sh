#!/bin/sh
# Stages mmh3 at 9c0652f for mmh3-job.sh, whose GPU job has no network and no Rust: its source with every crate
# vendored, and Rust 1.98, in the pilot folder's mmh3-9c0652f. Runs on BOTTOM's CPUs, never its GPU:
#   ssh top-menu "bottom shell" < docs/recipes/minimax-h3-cuda/mmh3-stage.sh
set -eu
stage=$HOME/Documents/drama3-pilot-20261014/mmh3-9c0652f
mkdir -p "$stage"
nice -n 10 docker run --rm --user 0 -v "$stage:/stage" gcc:12-bookworm sh -c '
set -eu
export RUSTUP_HOME=/stage/rustup CARGO_HOME=/stage/cargo
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain 1.98.0 --no-modify-path
rm -rf /stage/src && git init -q /stage/src && cd /stage/src
git fetch -q --depth 1 https://github.com/mmh3-org/mmh3 9c0652f4c39fd67ed2330aeafd61e95a1fe7e375
git checkout -q --detach FETCH_HEAD
mkdir -p .cargo
/stage/cargo/bin/cargo vendor --locked vendor >.cargo/config.toml
rm -rf /stage/cargo
chown -R 1000:1000 /stage'
du -sh "$stage"
