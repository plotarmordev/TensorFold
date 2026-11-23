#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
cache=${ZIG_LOCAL_CACHE_DIR:-$HOME/.cache/tensorfold/zig}
case "$cache" in /*) ;; *) echo "ZIG_LOCAL_CACHE_DIR must be absolute" >&2; exit 1;; esac
cache=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$cache")
case "$cache" in "$root"|"$root"/*) echo "ZIG_LOCAL_CACHE_DIR must be outside the checkout" >&2; exit 1;; esac
out=$cache/checkouts/$(printf '%s' "$root" | shasum -a 256 | cut -c1-16)/zig-out
[ "$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$out")" = "$out" ] || { echo "symlinked shared cache output refused" >&2; exit 1; }
mkdir -p "$out"
link() {
    if [ -L "$root/$1" ]; then
        [ "$(readlink "$root/$1")" = "$2" ] || { echo "different $1 link already exists" >&2; exit 1; }
    elif [ -e "$root/$1" ]; then
        echo "move the existing $1 out of the checkout first" >&2
        exit 1
    else
        ln -s "$2" "$root/$1"
    fi
}
link .zig-cache "$cache"
link zig-out "$out"
