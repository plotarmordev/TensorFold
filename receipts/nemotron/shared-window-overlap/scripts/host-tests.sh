#!/bin/sh
# Upstream's CI steps on TREE, a fresh cache dir each, then any extra STEPs; summary to stdout, logs as OUT.*.
# Usage: host-tests.sh TREE OUT [STEP...]
set -u
tree=$1 out=$(realpath -m "$2")
shift 2
zig=$HOME/opt/zig-aarch64-linux-0.17.0/zig
cd "$tree"
echo "$(git rev-parse HEAD) $(git status --short | tr '\n' ' ')(fresh --cache-dir for every step)"
for step in test-kernel-contract test test-golden fuzz-cortheon "$@"; do
    cache=$(mktemp -d /tmp/zc.XXXXXX)
    echo "== zig build $step"
    "$zig" build "$step" --cache-dir "$cache" --summary line >"$out.$step.log" 2>&1
    echo "exit=$?"
    grep -E "Build Summary|equal to the frozen|FAIL|passed|failed" "$out.$step.log" | tail -3
    rm -rf "$cache"
done
echo "== zig fmt --check zig build.zig"
"$zig" fmt --check zig build.zig
echo "exit=$?"
echo "== python3 tools/zig/lean_check.py"
python3 tools/zig/lean_check.py >"$out.lean.txt"
tail -1 "$out.lean.txt"
