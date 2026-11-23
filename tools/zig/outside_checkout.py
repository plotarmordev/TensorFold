"""Print a cache path's canonical form; refuse one that is the checkout or inside it.

Compares filesystem identities, not spellings, so a case variant on a case-insensitive disk is caught too.
"""
import os
from pathlib import Path
import sys


def outside(root, path):
    path = Path(path)
    if not path.is_absolute():
        sys.exit("ZIG_LOCAL_CACHE_DIR must be absolute")
    path = Path(os.path.realpath(path))
    root = os.stat(root)
    for folder in (path, *path.parents):
        if folder.exists() and os.path.samestat(os.stat(folder), root):
            sys.exit("ZIG_LOCAL_CACHE_DIR must be outside the checkout")
    return path


if __name__ == "__main__":
    print(outside(*sys.argv[1:]))
