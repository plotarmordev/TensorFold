#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
cache=$(python3 "$root/tools/zig/outside_checkout.py" "$root" "${ZIG_LOCAL_CACHE_DIR:-$HOME/.cache/tensorfold/zig}")
out=$cache/checkouts/$(printf '%s' "$root" | shasum -a 256 | cut -c1-16)/zig-out
[ "$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$out")" = "$out" ] || { echo "symlinked shared cache output refused" >&2; exit 1; }
mkdir -p "$out"
link() {
    if [ -e "$root/$1" ] && [ ! -L "$root/$1" ]; then
        echo "move the existing $1 out of the checkout first" >&2
        exit 1
    fi
    # -n: a concurrent run's fresh link is not followed, so no link lands inside the target;
    # a lost race leaves that link, any other failure is retried to show ln's own error.
    [ -L "$root/$1" ] || ln -sn "$2" "$root/$1" 2>/dev/null || [ -L "$root/$1" ] || ln -sn "$2" "$root/$1"
    # Compared in Python: command substitution would strip a trailing newline from the existing link's target.
    python3 -c 'import os, sys; sys.exit(os.readlink(sys.argv[1]) != sys.argv[2])' "$root/$1" "$2" ||
        { echo "different $1 link already exists" >&2; exit 1; }
}
link .zig-cache "$cache"
link zig-out "$out"
