#!/bin/sh
# One zig build STEP on TREE with a fresh cache, its full output to OUT. Usage: step.sh TREE OUT STEP
set -u
zig=$HOME/opt/zig-aarch64-linux-0.17.0/zig
c=$(mktemp -d /tmp/zc.XXXXXX)
cd "$1" || exit 1
{ echo "$(git rev-parse HEAD) $(git status --short | tr '\n' ' ')"; "$zig" build "$3" --cache-dir "$c" --summary all 2>&1; echo "exit=$?"; } >"$2"
rm -rf "$c"
tail -1 "$2"
