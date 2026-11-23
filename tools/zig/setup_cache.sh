#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cache=${ZIG_LOCAL_CACHE_DIR:-$HOME/.cache/tensorfold/zig}
mkdir -p "$cache"
cache=$(CDPATH= cd -- "$cache" && pwd -P)
if [ -L "$root/.zig-cache" ]; then
    [ "$(readlink "$root/.zig-cache")" = "$cache" ] || { echo "different .zig-cache link already exists" >&2; exit 1; }
elif [ -d "$root/.zig-cache" ]; then
    echo "move the existing .zig-cache out of the checkout first" >&2
    exit 1
elif [ -e "$root/.zig-cache" ]; then
    echo ".zig-cache exists and is not a directory" >&2
    exit 1
else
    ln -s "$cache" "$root/.zig-cache"
fi
