"""A card's scratch left loose in /tmp is found by its card number or its tool's hard-coded path, and no one else's is."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scratch_check

SINCE = 1_000_000


class Strays(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make(self, name, mtime=SINCE + 10):
        p = Path(self.tmp.name, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        os.utime(p, (mtime, mtime))
        return p

    def strays(self, uid=None):
        return scratch_check.strays(self.tmp.name, "245", SINCE, os.getuid() if uid is None else uid)

    def test_card_named_entries_made_since_the_card_started_are_strays(self):
        # the main agent's log, an A/B review copy's log and a helper's fetch inside a folder of its own
        want = [self.make("tf245-gpu.log"), self.make("review-245-a.log"), self.make("random-parent/tf245-helper.log")]
        self.assertEqual(sorted(want), self.strays())

    def test_other_cards_numbers_and_older_entries_are_not(self):
        for name in ("tf2450.log", "1245-bench.log", "card246.log", "receipt.txt", "a/b/c/tf245-too-deep.log"):
            self.make(name)
        self.make("tf245-old.log", mtime=SINCE - 1)
        self.assertEqual([], self.strays())

    def test_another_users_entry_is_not(self):
        self.make("tf245-gpu.log")
        self.assertEqual([], self.strays(uid=os.getuid() + 1))


class Main(unittest.TestCase):
    """main() run in a review worktree of card 245, against a /tmp of its own."""

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.tmp, self.root = Path(t.name, "tmp"), Path(t.name, "review-245-abc")
        self.tmp.mkdir()
        (self.root / "tools").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "tools/bench.py").write_text('LOCK = "/tmp/tensorfold-model.lock"\n')
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)

    def run_main(self):
        out = io.StringIO()
        with patch.object(scratch_check, "TMP", str(self.tmp)), contextlib.redirect_stdout(out), \
                self.assertRaises(SystemExit) as exit_:
            scratch_check.main()
        return exit_.exception.code, out.getvalue()

    def test_a_clean_review_worktree_passes(self):
        self.assertEqual((0, ""), self.run_main())

    def test_a_helpers_nested_fetch_fails_the_review_worktree(self):
        (self.tmp / "random-parent").mkdir()
        (self.tmp / "random-parent/tf245-helper.log").write_text("x")
        code, out = self.run_main()
        self.assertEqual(1, code)
        self.assertIn("random-parent/tf245-helper.log", out)

    def test_a_tool_writing_to_a_fixed_tmp_folder_fails(self):
        # the former needle bench layout: /tmp/nem_needle carries no card number, so the source line is what shows it
        (self.root / "tools/needle.py").write_text('d = "/tmp/nem_needle"\n')
        code, out = self.run_main()
        self.assertEqual(1, code)
        self.assertIn("tools/needle.py:1:", out)


if __name__ == "__main__":
    unittest.main()
