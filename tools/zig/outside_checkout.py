"""Print a cache path's canonical form; refuse one that is the checkout or inside it.

Compares filesystem identities, not spellings, so a case variant on a case-insensitive disk is caught too.
"""
import os
from pathlib import Path
import sys


def outside(root, path):
    # Shell command substitution strips trailing newlines, so the callers would use another path than the one checked.
    if "\n" in str(path):
        sys.exit("ZIG_LOCAL_CACHE_DIR must not contain a newline")
    path = Path(path)
    if not path.is_absolute():
        sys.exit("ZIG_LOCAL_CACHE_DIR must be absolute")
    path = Path(os.path.realpath(path))
    root = os.stat(root)
    for folder in (path, *path.parents):
        if folder.exists() and os.path.samestat(os.stat(folder), root):
            sys.exit("ZIG_LOCAL_CACHE_DIR must be outside the checkout")
    return path


def shared_cache(root):
    """The checked shared cache: ZIG_LOCAL_CACHE_DIR, else $HOME/.cache/tensorfold/zig."""
    return outside(root, os.environ.get("ZIG_LOCAL_CACHE_DIR") or Path.home() / ".cache/tensorfold/zig")


if __name__ == "__main__":
    print(outside(*sys.argv[1:]))
