"""The needle bench writes its token files and reports into its own checkout's .scratch, not /tmp."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
# Byte-per-token stand-in for the tokenizers package, so the bench runs without a model.
TOKENIZERS = """class Tokenizer:
    @staticmethod
    def from_file(_):
        return Tokenizer()
    def encode(self, s):
        return type("E", (), {"ids": list(s.encode())})()
    def decode(self, ids):
        return bytes(ids).decode()
"""


class Destination(unittest.TestCase):
    def test_token_files_go_to_the_checkouts_scratch(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            tools = t / "repo/tools/zig"
            tools.mkdir(parents=True)
            shutil.copy(HERE / "xpu_nem_needle.py", tools)
            (tools / "xpu_guard.sh").write_text("#!/bin/sh\nexit 0\n")
            (tools / "xpu_guard.sh").chmod(0o755)
            (t / "stub").mkdir()
            (t / "stub/tokenizers.py").write_text(TOKENIZERS)
            (t / "ck").mkdir()
            (t / "ck/config.json").write_text('{"eos_token_id": 0}')
            (t / "home/eval/wikitext-2-raw").mkdir(parents=True)
            (t / "home/eval/wikitext-2-raw/wiki.train.raw").write_text("haystack " * 100)
            env = dict(os.environ, HOME=str(t / "home"), PYTHONPATH=str(t / "stub"))
            run = subprocess.run([sys.executable, str(tools / "xpu_nem_needle.py"), "--ckpt", str(t / "ck"),
                                  "--depths", "50", "400"], env=env, capture_output=True, text=True)
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertEqual(["n_400_50.txt"], os.listdir(t / "repo/.scratch/nem_needle"))


if __name__ == "__main__":
    unittest.main()
