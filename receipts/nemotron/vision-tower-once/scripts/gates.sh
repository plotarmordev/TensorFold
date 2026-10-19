#!/bin/sh
# Upstream's gates for the vision fix, CPU only (no GPU process): CI steps, fmt, lean against main, native build.
# Usage: gates.sh DIR (holding base/ and tip/ checkouts); logs to DIR/out/.
set -u
dir=$(realpath "$1")
out=$dir/out
mkdir -p "$out"
zig=$HOME/opt/zig-aarch64-linux-0.17.0/zig
nvcc=/usr/local/cuda/bin/nvcc
step() { # step TREE LOG ARGS...: one zig build in TREE on a fresh cache (every test runs), its exit and summary
    tree=$1 log=$2
    shift 2
    cache=$(mktemp -d)
    echo "== [$tree] zig build $*"
    (cd "$dir/$tree" && "$zig" build "$@" --cache-dir "$cache" -j1 --summary line) >"$out/$log" 2>&1
    echo "exit=$?"
    rm -rf "$cache"
    grep -E "Build Summary|equal to the frozen|expected|found" "$out/$log" | tail -3
}
for t in base tip; do echo "$t $(git -C "$dir/$t" rev-parse HEAD) $(git -C "$dir/$t" status --short | tr '\n' ' ')"; done
for s in test-kernel-contract test test-golden fuzz-cortheon; do step tip "tip.$s.log" "$s"; done
step base base.test.log test
for t in base tip; do
    echo "== [$t] zig fmt --check zig build.zig"
    (cd "$dir/$t" && "$zig" fmt --check zig build.zig)
    echo "exit=$?"
    echo "== [$t] python3 tools/zig/lean_check.py"
    (cd "$dir/$t" && python3 tools/zig/lean_check.py) >"$out/$t.lean.txt"
    tail -1 "$out/$t.lean.txt"
done
echo "== lean_new base -> tip"
python3 "$dir/tip/tools/zig/lean_new.py" "$out/base.lean.txt" "$out/tip.lean.txt" >"$out/lean-new.txt" 2>&1
echo "exit=$?"
cat "$out/lean-new.txt"
common="-Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline"
release="-Doptimize=fast -Dstrip=true"
step tip tip.fatbins.log fatbins $common "-Dnvcc=$nvcc" -Dsm=121 --prefix "$out/build"
step tip tip.native.log native $common $release "-Dfatbins=$out/build/fatbin" --prefix "$out/build"
(cd "$out/build" && sha256sum native/bin/tensorfold-native)
