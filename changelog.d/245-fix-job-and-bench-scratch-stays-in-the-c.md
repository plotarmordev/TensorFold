# Card scratch covers copies, helpers and the needle bench

- `AGENTS.md` names the scratch folder from the worktree root, so a subdirectory command lands there too. Review and A/B
  checkout copies go under it as `git worktree` checkouts removed when the step ends. Every delegated helper's brief
  carries the rule.
- `tools/scratch_check.py` exits 1 while `/tmp` holds an entry named for the card (up to three folders deep, so a
  helper's own folder counts) made since its worktree was, or while a file under `tools/` names a `/tmp` path other than
  the shared locks. It reads `review-` worktrees too; agents run it after each helper's task and at the end of the step.
- `tools/zig/xpu_nem_needle.py` writes its token files and reports to the checkout's `.scratch/nem_needle`, not
  `/tmp/nem_needle`. The lean base checkout in `docs/pr/SKILL.md` and the weight-read results and fixtures in
  `tools/zig/weight_reads.md` moved into `.scratch/` as well.
