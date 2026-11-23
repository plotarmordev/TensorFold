"""List a card's scratch left loose in /tmp; exit 1 when there is any (AGENTS.md, "Keep scratch in the worktree").

A stray is a top-level /tmp entry of this user, changed since the card's worktree was made, whose name carries the
card number (the worktree folder's leading digits) as a whole number: `tf245-gpu.log`, `review-245-a.log`.
Run it from the worktree after the step and after every helper's task, since helpers share the user and the /tmp.
"""
import os
import re
import subprocess
import sys
from pathlib import Path


def strays(tmp, card, since, uid):
    """Entries in tmp named for card, owned by uid and changed at or after since."""
    pat = re.compile(rf"(?<!\d){card}(?!\d)")
    found = []
    for e in os.scandir(tmp):
        st = e.stat(follow_symlinks=False)
        if pat.search(e.name) and st.st_uid == uid and st.st_mtime >= since:
            found.append(Path(tmp, e.name))
    return sorted(found)


def main():
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               check=True).stdout.strip())
    card = re.match(r"\d+", root.name)
    if not card:
        sys.exit(f"{root.name}: the worktree folder does not start with a card number")
    st = (root / ".git").stat()
    found = strays("/tmp", card.group(), getattr(st, "st_birthtime", st.st_mtime), os.getuid())
    for p in found:
        print(f"stray: {p} (move it into {root / '.scratch'} or delete it)")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
