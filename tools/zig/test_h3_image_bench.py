#!/usr/bin/env python3
"""bench-script.sh's summary from fixture logs in the three CUDA tools' formats, without a GPU."""
import tempfile
import unittest
from pathlib import Path

import h3_image_bench as bench

PROMPT = """24 tokens; text encoder, 50 layers: 55.30 s wall, 53.20 s GPU
condition_proj and the token refiner: 1.60 s wall, 1.50 s GPU
wrote /work/out/x.safetensors: text [24, 4096] bf16, text_tags and ids
prompt: 57.30 s
"""
WARM_IMAGE = """2048x2048 canvas, 16408 rows (text 24, references 0, target 16384), 1 tiles, 60 blocks, 6 steps, case ready in 1.2 s
step 1/6: 6.50 s
transformer: 26.50 s
"""
IMAGE = "model loaded in 88.7 s\n" + WARM_IMAGE
DECODER = """decoder weights in bf16 in 4.6 s
2048x2048 canvas: decoded 25 tiles in 3.10 s (2.80 s on the GPU), 7.70 s in all; wrote /work/out/x.png at 2048x2048
"""


class Summary(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        (self.out / "settings").write_text("commit: abc123\nseed: 42\n")
        (self.out / "selftests").write_text("transformer PASS\ndecoder FAIL\nprompt PASS\n")

    def case(self, name, start, end, smi):
        case = self.out / name
        Path(f"{case}.time").write_text(f"{start}\n{end}\n")
        Path(f"{case}.prompt.log").write_text(PROMPT)
        Path(f"{case}.image.log").write_text(IMAGE if name.endswith("-1") else WARM_IMAGE)
        Path(f"{case}.decoder.log").write_text(DECODER)
        Path(f"{case}.smi").write_text(smi)

    def test_stage_times_peak_memory_power_and_energy(self):
        self.case("2048-1", 100.0, 300.0, "100.0, 2000, 50.0, 1000+500\n100.5, 4000, 70.0, 3000\n")
        run = bench.measure(self.out / "2048-1")
        self.assertEqual(run["wall s"], 200.0)
        self.assertEqual((run["prompt s"], run["load s"], run["transformer s"], run["decoder s"]), (57.3, 88.7, 26.5, 7.7))
        self.assertEqual(run["steps"], 6)
        self.assertEqual(run["peak MiB"], 4000)
        self.assertEqual(run["avg W"], 60.0)
        self.assertEqual(run["kJ"], 12.0)

    def test_unified_memory_falls_back_to_the_processes(self):
        smi = "1.0, [N/A], 40.0, 30000+2000\n1.5, [N/A], [N/A], \n2.0, [N/A], 60.0, 50000\n"
        self.case("2048-1", 0.0, 10.0, smi)
        run = bench.measure(self.out / "2048-1")
        self.assertEqual(run["peak MiB"], 50000)
        self.assertEqual(run["avg W"], 50.0)

    def test_warm_runs_on_the_resident_model_have_no_load(self):
        self.case("2048-2", 0.0, 40.0, "")
        run = bench.measure(self.out / "2048-2")
        self.assertIsNone(run["load s"])
        self.assertEqual((run["transformer s"], run["steps"]), (26.5, 6))

    def test_a_failed_run_keeps_what_finished(self):
        case = self.out / "4096-1"
        Path(f"{case}.time").write_text("5.0\n")
        Path(f"{case}.prompt.log").write_text(PROMPT)
        run = bench.measure(case)
        self.assertEqual(run["prompt s"], 57.3)
        for column in ("wall s", "load s", "decoder s", "steps", "peak MiB", "avg W", "kJ"):
            self.assertIsNone(run[column], column)

    def test_summary_records_selftests_cold_warm_and_warm_mean(self):
        self.case("2048-1", 0.0, 200.0, "0, 1000, 50.0, \n")
        self.case("2048-2", 0.0, 100.0, "0, 1000, 50.0, \n")
        self.case("2048-3", 0.0, 120.0, "0, 1000, 50.0, \n")
        text = bench.summary(self.out)
        self.assertIn("- commit: abc123", text)
        self.assertIn("- decoder FAIL", text)
        self.assertIn("| 2048² cold | 200.0 | 57.3 | 88.7 | 26.5 | 7.7 | 6 | 1000.0 | 50.0 | 10.0 |", text)
        self.assertIn("| 2048² warm 2 | 120.0 |", text)
        self.assertIn("| 2048² warm mean | 110.0 | 57.3 | — | 26.5 | 7.7 | 6.0 | 1000.0 | 50.0 | 5.5 |", text)

    def test_a_failed_or_missing_selftest_marks_the_timings_as_not_the_sparks(self):
        self.assertIn("**Not the Spark's result: the decoder selftest failed;", bench.summary(self.out))
        (self.out / "selftests").write_text("transformer PASS\ndecoder PASS\nprompt PASS\n")
        text = bench.summary(self.out)
        self.assertIn("Every selftest passed.", text)
        self.assertNotIn("Not the Spark's result", text)
        (self.out / "selftests").unlink()
        self.assertIn("**Not the Spark's result: no selftest ran.**", bench.summary(self.out))

    def test_a_missing_number_in_a_warm_run_leaves_the_mean_empty(self):
        self.case("2048-1", 0.0, 200.0, "0, 1000, 50.0, \n")
        self.case("2048-2", 0.0, 100.0, "")
        self.case("2048-3", 0.0, 120.0, "0, 1000, 50.0, \n")
        self.assertIn("| 2048² warm mean | 110.0 | 57.3 | — | 26.5 | 7.7 | 6.0 | — | — | — |", bench.summary(self.out))


if __name__ == "__main__":
    unittest.main()
