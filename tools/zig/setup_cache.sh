#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cache=${ZIG_LOCAL_CACHE_DIR:-$HOME/.cache/tensorfold/zig}
out=$cache/checkouts/$(printf '%s' "$root" | shasum -a 256 | cut -c1-16)/zig-out
mkdir -p "$out"
cache=$(CDPATH= cd -- "$cache" && pwd -P)
out=$(CDPATH= cd -- "$out" && pwd -P)
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
