#!/usr/bin/env python3
"""Per node, build and case: each mode's median wave tok/s (on time, late) and the branch against the base in the same mode.

  modes.py speed/a speed/b

A wave is late when request 2's first text trails request 1's by more than 1.5 admission steps (request 3 after
request 2); a wave of 2 has no third request, so it is late past 130 ms, 1.5 times the 86-92 ms step every larger
wave here shows. Only measured waves count (not the warmup).
"""
import json
import statistics
import sys
from pathlib import Path


def late(wave: dict) -> bool:
    f = wave["first_ms"]
    return f[1] - f[0] > (1.5 * (f[2] - f[1]) if len(f) > 2 else 130)


def main():
    for root in map(Path, sys.argv[1:]):
        split = {}
        for jl in sorted(root.glob("*.waves.jsonl")):
            build = jl.name.removesuffix(".waves.jsonl").split("-", 1)[1]
            for w in map(json.loads, jl.read_text().splitlines()):
                if not w["warmup"]:
                    split.setdefault((w["case"], w["n"]), {}).setdefault(build, ([], []))[late(w)].append(w)
        print(f"\n{root}\n")
        print("| case | base on time / late (waves) | branch on time / late (waves) | branch vs base, same mode |")
        print("|---|---|---|---|")
        for (case, n), builds in sorted(split.items()):
            med = {b: [statistics.median(x["wave_tps"] for x in m) if m else None for m in modes] for b, modes in builds.items()}
            count = {b: f"{len(modes[0])}/{len(modes[1])}" for b, modes in builds.items()}

            def cell(b):
                return " / ".join("—" if v is None else f"{v:.2f}" for v in med[b]) + f" ({count[b]})"

            gain = " / ".join("—" if a is None or b is None else f"{(b / a - 1) * 100:+.1f} %"
                              for a, b in zip(med["base"], med["branch"]))
            print(f"| {case} c{n} | {cell('base')} | {cell('branch')} | {gain} |")


if __name__ == "__main__":
    main()
