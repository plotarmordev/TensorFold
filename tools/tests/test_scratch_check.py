"""A card's scratch left loose in /tmp is found by its card number, and no one else's /tmp entry is."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scratch_check

SINCE = 1_000_000


class Strays(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make(self, name, mtime=SINCE + 10):
        p = Path(self.tmp.name, name)
        p.mkdir() if "." not in name else p.write_text("x")
        os.utime(p, (mtime, mtime))
        return p

    def strays(self, uid=None):
        return scratch_check.strays(self.tmp.name, "245", SINCE, os.getuid() if uid is None else uid)

    def test_card_named_entries_made_since_the_card_started_are_strays(self):
        # a helper's GPU fetch, the main agent's log and an A/B review copy all count
        want = [self.make("tf245-gpu.log"), self.make("review-245-a.log"), self.make("ab_245")]
        self.assertEqual(sorted(want), self.strays())

    def test_other_cards_numbers_and_older_entries_are_not(self):
        for name in ("tf2450.log", "1245-bench.log", "card246.log", "receipt.txt"):
            self.make(name)
        self.make("tf245-old.log", mtime=SINCE - 1)
        self.assertEqual([], self.strays())

    def test_another_users_entry_is_not(self):
        self.make("tf245-gpu.log")
        self.assertEqual([], self.strays(uid=os.getuid() + 1))


if __name__ == "__main__":
    unittest.main()
