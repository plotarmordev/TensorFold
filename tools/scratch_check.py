"""List a card's scratch left loose in /tmp; exit 1 when there is any (AGENTS.md, "Keep scratch in the worktree").

A stray is an entry of this user up to three folders deep in /tmp, changed since the card's worktree was made, whose
name carries the card number (the worktree folder's leading digits, after any `review-`) as a whole number:
`tf245-gpu.log`, `some-helper-dir/review-245-a.log`. Output with no card number in its name is caught at the source:
a tool line naming a /tmp path other than the shared locks and stop files is reported too, since TMPDIR cannot move it.
Run it from the worktree after the step and after every helper's task, since helpers share the user and the /tmp.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

TMP = "/tmp"
DEPTH = 3
# Shared locks and stop files that live in /tmp on purpose.
SHARED = {"/tmp/arc_stop", "/tmp/b70.lock", "/tmp/tensorfold-model.lock", "/tmp/tensorfold-speech.lock"}
TMP_PATH = re.compile(r"/tmp/[A-Za-z0-9_.-]+")  # git grep -E knows no \w


def strays(tmp, card, since, uid, depth=DEPTH):
    """Entries under tmp named for card, owned by uid and changed at or after since; a match is not searched inside."""
    pat = re.compile(rf"(?<!\d){card}(?!\d)")
    found = []
    for e in os.scandir(tmp):
        try:
            st = e.stat(follow_symlinks=False)
            if st.st_uid != uid:
                continue
            if pat.search(e.name):
                if st.st_mtime >= since:
                    found.append(Path(e.path))
            elif depth > 1 and e.is_dir(follow_symlinks=False):
                found += strays(e.path, card, since, uid, depth - 1)
        except OSError:  # gone or unreadable: other agents churn /tmp while it is scanned
            continue
    return sorted(found)


def hardcoded(root):
    """Tool lines (tests aside) that name a /tmp path other than the shared locks and stop files."""
    out = subprocess.run(["git", "grep", "--untracked", "-nE", TMP_PATH.pattern, "--", "tools", ":!*test_*",
                          ":!*_test.py"], cwd=root, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if set(TMP_PATH.findall(line)) - SHARED]


def main():
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               check=True).stdout.strip())
    card = re.match(r"(?:review-)?(\d+)", root.name)
    if not card:
        sys.exit(f"{root.name}: the worktree folder does not start with a card number")
    st = (root / ".git").stat()
    found = strays(TMP, card.group(1), getattr(st, "st_birthtime", st.st_mtime), os.getuid())
    for p in found:
        print(f"stray: {p} (move it into {root / '.scratch'} or delete it)")
    lines = hardcoded(root)
    for line in lines:
        print(f"hard-coded /tmp output: {line} (write it under the worktree's .scratch)")
    sys.exit(1 if found or lines else 0)


if __name__ == "__main__":
    main()
