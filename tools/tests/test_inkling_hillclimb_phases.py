"""hillclimb.sh logs the build and the measurement as separate phases; only the measurement runs under its timeout."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

RECIPE = Path(__file__).resolve().parents[2] / "docs/recipes/inkling-2-mini-cuda"


class Phases(unittest.TestCase):
    def test_a_slow_setup_leaves_the_measurement_its_whole_allowance(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipe = root / "docs/recipes/inkling-2-mini-cuda"
            recipe.mkdir(parents=True)
            shutil.copy(RECIPE / "hillclimb.sh", recipe)
            # a slow build, then a measurement that fails at its first step
            (recipe / "setup.sh").write_text("sleep 2\nmodel=/m\nout=$PWD/out\n: >setup-done\n")
            (recipe / "ceiling.py").write_text("raise SystemExit(3)\n")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            # records what the deadline wraps and whether setup had finished, then runs it
            (bin_dir / "timeout").write_text(
                '#!/bin/sh\necho "$* setup-done=$([ -f setup-done ] && echo yes)" >timeout-args\n'
                'shift 3\nexec "$@"\n')
            (bin_dir / "timeout").chmod(0o755)
            run = subprocess.run(["sh", str(recipe / "hillclimb.sh")], cwd=root, capture_output=True, text=True,
                                 env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
            self.assertEqual(3, run.returncode, run.stderr)
            build, measure = run.stdout.splitlines()
            self.assertGreaterEqual(int(build.removeprefix("build: ").removesuffix(" s")), 2)
            self.assertLess(int(measure.removeprefix("measure: ").removesuffix(" s")), 2)
            self.assertEqual(f"--signal=INT --kill-after=20s 580 sh {recipe / 'hillclimb.sh'} measure setup-done=yes",
                             (root / "timeout-args").read_text().strip())


if __name__ == "__main__":
    unittest.main()
