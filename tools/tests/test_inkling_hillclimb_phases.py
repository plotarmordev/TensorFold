"""hillclimb.sh runs the build outside its deadlines, then each measurement job under its own 580 s timeout."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

RECIPE = Path(__file__).resolve().parents[2] / "docs/recipes/inkling-2-mini-cuda"
CAP = "--signal=INT --kill-after=20s 580 sh {} measure"


class Phases(unittest.TestCase):
    def run_hillclimb(self, root, timeout_body):
        recipe = root / "docs/recipes/inkling-2-mini-cuda"
        recipe.mkdir(parents=True)
        shutil.copy(RECIPE / "hillclimb.sh", recipe)
        # a slow build that marks when it has finished
        (recipe / "setup.sh").write_text("sleep 2\nmodel=/m\nout=$PWD/out\nmkdir -p out\n: >setup-done\n")
        bin_dir = root / "bin"
        bin_dir.mkdir()
        # records each deadline's arguments and whether setup had finished, then acts as told
        (bin_dir / "timeout").write_text(
            '#!/bin/sh\necho "$* setup-done=$([ -f setup-done ] && echo yes)" >>timeout-args\n' + timeout_body)
        (bin_dir / "timeout").chmod(0o755)
        run = subprocess.run(["sh", str(recipe / "hillclimb.sh")], cwd=root, capture_output=True, text=True,
                             env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
        return run, CAP.format(recipe / "hillclimb.sh"), (root / "timeout-args").read_text().splitlines()

    def test_a_slow_setup_leaves_the_measurement_its_whole_allowance(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            # the ceiling job fails at once
            run, cap, calls = self.run_hillclimb(root, "exit 3\n")
            self.assertEqual(3, run.returncode, run.stderr)
            build, measure = run.stdout.splitlines()
            self.assertGreaterEqual(int(build.removeprefix("build: ").removesuffix(" s")), 2)
            self.assertLess(int(measure.removeprefix("measure ceiling: ").removesuffix(" s")), 2)
            self.assertEqual([f"{cap} ceiling setup-done=yes"], calls)

    def test_each_arm_gets_its_own_deadline_and_a_cut_off_arm_is_ledgered(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            # the first two jobs finish inside their deadline, the lever arm runs out of its own
            run, cap, calls = self.run_hillclimb(root, 'case "$*" in *warps-1*) exit 124;; esac\n')
            self.assertEqual(124, run.returncode, run.stderr)
            self.assertEqual([f"{cap} ceiling setup-done=yes", f"{cap} control-1 setup-done=yes",
                              f"{cap} warps-1 TF_NVFP4_DECODE_WARPS=1 setup-done=yes"], calls)
            self.assertEqual('{"arm":"warps-1","failed":"cut off at 580 s"}\n',
                             (root / "out/ledger.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
