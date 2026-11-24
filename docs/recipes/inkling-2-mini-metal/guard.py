#!/usr/bin/env python3
"""bench.sh's void-run gate: a bench_concurrent report must have clean cells at plausible rates.

Usage: guard.py REPORT [PARTNER]. Environment: MIN_TPS is the aggregate tok/s floor each cell must
reach (default 0.3, below main's 0.85-0.9 control baseline; the candidate takes ~20). A cell with
failed or unmeasured requests fails whatever its rate: bench_concurrent reports a positive
aggregate from the streams that did finish. PARTNER is the first control's report; a control more
than 2x below its partner fails. Exit 0 is clean, non-zero is a void run.
"""
import json
import os
import sys

FLOOR = 0.3
RATIO = 2


def report(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        sys.exit(f"GUARD: no report at {path}; the run did not finish its cells")


def rates(rep):
    return {(c["prompt"], c["streams"]): c.get("aggregate_tps") for c in rep.get("cells", [])}


def check(path, floor, partner_path=None):
    rep = report(path)
    for cell in rep.get("cells", []):
        key = (cell["prompt"], cell["streams"])
        if cell.get("failed") or cell.get("unmeasured"):
            sys.exit(f"GUARD {path}: cell {key} has {cell.get('failed', 0)} failed and "
                     f"{cell.get('unmeasured', 0)} unmeasured requests")
        # A run whose concurrent, alone or serial outputs disagree was not measuring the engine.
        for check_name in ("alone", "serial"):
            verdict = cell.get(check_name)
            if verdict and (verdict.get("unequal") or verdict.get("failed")):
                sys.exit(f"GUARD {path}: cell {key} {check_name} comparisons: "
                         f"{verdict.get('unequal', 0)} unequal, {verdict.get('failed', 0)} failed")
        rate = cell.get("aggregate_tps")
        if rate is None or rate < floor:
            sys.exit(f"GUARD {path}: cell {key} at {rate} tok/s, below floor {floor}")
    if partner_path:
        theirs = rates(report(partner_path))
        for key, rate in rates(rep).items():
            other = theirs.get(key)
            if rate is not None and other and rate < other / RATIO:
                sys.exit(f"GUARD {path}: cell {key} at {rate} tok/s, over 2x below partner's {other}")


if __name__ == "__main__":
    check(sys.argv[1], float(os.environ.get("MIN_TPS", FLOOR)),
          sys.argv[2] if len(sys.argv) > 2 else None)
