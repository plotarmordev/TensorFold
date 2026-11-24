"""The inkling-2-mini-metal bench guard: clean cells at plausible rates pass, void runs fail."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "docs" / "recipes" / "inkling-2-mini-metal"))
import guard  # noqa: E402


def report(cells):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"cells": cells}, tmp)
    tmp.close()
    return tmp.name


def cell(streams=1, tps=60.0, failed=0, unmeasured=0):
    return {"prompt": "mixed", "streams": streams, "aggregate_tps": tps,
            "failed": failed, "unmeasured": unmeasured}


class GuardTests(unittest.TestCase):
    def test_clean_cells_pass(self):
        guard.check(report([cell(1), cell(4, 3.2)]), floor=0.3)

    def test_positive_rate_with_failed_requests_fails(self):
        # bench_concurrent aggregates only the streams that finished; a partial cell still reports a rate.
        with self.assertRaises(SystemExit):
            guard.check(report([cell(1, 55.0, failed=1)]), floor=0.3)

    def test_positive_rate_with_unmeasured_requests_fails(self):
        with self.assertRaises(SystemExit):
            guard.check(report([cell(4, 200.0, unmeasured=2)]), floor=0.3)

    def test_below_floor_fails(self):
        with self.assertRaises(SystemExit):
            guard.check(report([cell(1, 0.2)]), floor=0.3)

    def test_missing_report_fails(self):
        with self.assertRaises(SystemExit):
            guard.check("/nonexistent/report.json", floor=0.3)

    def test_control_over_twice_below_partner_fails(self):
        partner = report([cell(1, 0.9), cell(4, 3.6)])
        with self.assertRaises(SystemExit):
            guard.check(report([cell(1, 0.4), cell(4, 3.4)]), floor=0.1, partner_path=partner)

    def test_control_within_partner_ratio_passes(self):
        partner = report([cell(1, 0.9), cell(4, 3.6)])
        guard.check(report([cell(1, 0.85), cell(4, 3.3)]), floor=0.1, partner_path=partner)


if __name__ == "__main__":
    unittest.main()
