"""drama3_limits_gate.sh: only d3_router_compact's pending record may make the limits audit exit non-zero."""
import subprocess
import tempfile
import unittest
from pathlib import Path

GATE = Path(__file__).with_name("drama3_limits_gate.sh")
ROUTER = "PASS limits sm_{sm} drama3 d3_router_compact: 30 registers, 192 static shared bytes, bound 256 (record stale)"
RECORD = "RECORD drama3 d3_router_compact: .usage = .{ }"


def gate(status: int, *lines: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".log") as log:
        log.write("\n".join(lines) + "\n")
        log.flush()
        return subprocess.run(["sh", str(GATE), str(status), log.name]).returncode


class LimitsGateTests(unittest.TestCase):
    def test_clean_audit_passes(self):
        self.assertEqual(gate(0, "PASS kernel-contract-limits: every registered launch fits"), 0)

    def test_pending_router_record_alone_passes(self):
        sms = (80, 86, 89, 120, 121)
        self.assertEqual(gate(1, *(ROUTER.format(sm=sm) for sm in sms), RECORD, "FAIL kernel-contract-limits: 5 problems"), 0)

    def test_failed_launch_with_record_fails(self):
        failed = "FAIL limits sm_121 drama3 d3_router_compact: 300 registers, 0 static shared bytes, bound 256 (record stale): "
        self.assertNotEqual(gate(1, failed + "too many registers", RECORD, "FAIL kernel-contract-limits: 1 problems"), 0)

    def test_missed_control_fails(self):
        missed = "MISSED limits sm_121 control: oversized launch"
        self.assertNotEqual(gate(1, ROUTER.format(sm=121), missed, RECORD, "FAIL kernel-contract-limits: 2 problems"), 0)

    def test_other_stale_record_fails(self):
        other = "PASS limits sm_121 drama3 d3_rmsnorm: 20 registers, 144 static shared bytes, bound 0 (record stale)"
        self.assertNotEqual(gate(1, ROUTER.format(sm=121), other, RECORD, "FAIL kernel-contract-limits: 2 problems"), 0)

    def test_error_without_summary_fails(self):
        self.assertNotEqual(gate(1, ROUTER.format(sm=121), RECORD), 0)


if __name__ == "__main__":
    unittest.main()
