#!/usr/bin/env python3
"""The price bar of the k-th draft at c4 on a cost table: the least acceptance at depth k for which budgets (sim.py's
allocate, batched head steps, no overhead) grants it to every stream, when depths 1 .. k-1 are verified at 1.0 and
every deeper depth sits at 0.5.

Usage: bars.py TABLE.bin"""
import sys

import sim

STREAMS = 4


def granted(cost, level, k, x):
    rates = [1.0] * (k - 1) + [x] + [0.5] * (len(sim.PRIOR) - k)
    counts = sim.allocate([sim.chain(rates, sim.BATCH_ROWS - 1)] * STREAMS, cost, sim.BATCH_ROWS, level, 0.0)
    return all(c >= k for c in counts)


def main():
    cost, level = sim.load(sys.argv[1])
    for k in (5, 6, 7):
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if granted(cost, level, k, mid) else (mid, hi)
        print(f"draft {k}: bar {hi:.3f} (prior {sim.PRIOR[k - 1]:.2f})")


if __name__ == "__main__":
    main()
