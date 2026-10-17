#!/bin/sh
# Upstream's CI steps on TREE, a fresh cache dir each; summary to stdout, full logs and lean output as OUT.*. Usage: host-tests.sh TREE OUT
set -u
tree=$1 out=$(realpath -m "$2")
zig=$HOME/opt/zig-aarch64-linux-0.17.0/zig
cd "$tree"
echo "$(git rev-parse HEAD) (fresh --cache-dir for every step)"
for step in test-kernel-contract test test-golden fuzz-cortheon; do
    cache=$(mktemp -d <tmp>/zc.XXXXXX)
    echo "== zig build $step"
    "$zig" build "$step" --cache-dir "$cache" --summary line >"$out.$step.log" 2>&1
    echo "exit=$?"
    grep -E "Build Summary|equal to the frozen|FAIL" "$out.$step.log" | tail -3
    rm -rf "$cache"
done
echo "== zig fmt --check zig build.zig"
"$zig" fmt --check zig build.zig
echo "exit=$?"
echo "== python3 tools/zig/lean_check.py"
python3 tools/zig/lean_check.py >"$out.lean.txt"
tail -1 "$out.lean.txt"
