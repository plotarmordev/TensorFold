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
# Stands in for xpu_guard.sh and tensorfold-xpu run: writes the report the bench reads, answering with the code.
GUARD = f"""#!{sys.executable}
import json, sys
a = sys.argv
json.dump({{"tokens": list(b" 7391-ALPHA."), "prefill_seconds": 1.0, "ms_per_token": 10.0}},
          open(a[a.index("--report") + 1], "w"))
"""


class Destination(unittest.TestCase):
    def test_token_files_and_reports_go_to_the_checkouts_scratch(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            tools = t / "repo/tools/zig"
            tools.mkdir(parents=True)
            shutil.copy(HERE / "xpu_nem_needle.py", tools)
            (tools / "xpu_guard.sh").write_text(GUARD)
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
            self.assertRegex(run.stdout, r"400\s+50%\s+FOUND")
            self.assertEqual(["n_400_50.txt", "report_400_50.json"], sorted(os.listdir(t / "repo/.scratch/nem_needle")))


if __name__ == "__main__":
    unittest.main()
