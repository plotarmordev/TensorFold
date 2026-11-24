# Card scratch stays in the card's worktree

- Agents put GPU job logs and fetches, bench outputs, tars, logs and nsys reports in a gitignored `.scratch/` at the
  worktree root, not loose in `/tmp` (`AGENTS.md`). The folder goes when the card's worktree is removed on done or
  drop.
