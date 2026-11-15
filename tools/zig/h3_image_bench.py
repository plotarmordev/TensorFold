#!/usr/bin/env python3
"""SUMMARY.md for docs/recipes/minimax-h3-cuda/bench-script.sh's output folder: every run's stage times from the
tools' logs, and its peak memory, average power and energy from the nvidia-smi samples taken during it."""
import re
import statistics
import sys
from pathlib import Path

STAGES = {  # column: (log, pattern of its seconds)
    "prompt": ("prompt", r"^prompt: ([\d.]+) s$"),
    "load": ("image", r"loaded in ([\d.]+) s$"),
    "transformer": ("image", r"^transformer: ([\d.]+) s$"),
    "decoder": ("decoder", r"([\d.]+) s in all;"),
}
COLUMNS = ("wall s", "prompt s", "load s", "transformer s", "decoder s", "steps", "peak MiB", "avg W", "kJ")


def found(path, pattern):
    """The first group of pattern's first match in path, or None when the file or the line is missing."""
    if not path.exists():
        return None
    match = re.search(pattern, path.read_text(errors="replace"), re.MULTILINE)
    return match and match.group(1)


def number(text):
    """A sample's field as a float; nvidia-smi writes [N/A] or [Not Supported] where it has none."""
    try:
        return float(text)
    except ValueError:
        return None


def samples(path):
    """(memory MiB, watts) per sample. The Spark's unified memory reports no memory.used: then the compute processes'."""
    rows = []
    for line in path.read_text().splitlines() if path.exists() else ():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        apps = [number(app) for app in fields[3].split("+") if app]
        memory = number(fields[1])
        if memory is None and apps and None not in apps:
            memory = sum(apps)
        rows.append((memory, number(fields[2])))
    return rows


def measure(case):
    """One run's numbers by column, None where it has none."""
    run = {}
    times = Path(f"{case}.time").read_text().split() if Path(f"{case}.time").exists() else []
    run["wall s"] = float(times[1]) - float(times[0]) if len(times) == 2 else None
    for column, (log, pattern) in STAGES.items():
        value = found(Path(f"{case}.{log}.log"), pattern)
        run[f"{column} s"] = value and float(value)
    steps = found(Path(f"{case}.image.log"), r"(\d+) steps, loaded in")
    run["steps"] = steps and int(steps)
    rows = samples(Path(f"{case}.smi"))
    memory = [m for m, _ in rows if m is not None]
    watts = [w for _, w in rows if w is not None]
    run["peak MiB"] = max(memory) if memory else None
    run["avg W"] = statistics.fmean(watts) if watts else None
    run["kJ"] = run["avg W"] * run["wall s"] / 1000 if watts and run["wall s"] is not None else None
    return run


def cell(value):
    if value is None:
        return "—"
    return str(value) if isinstance(value, int) else f"{value:.1f}"


def row(label, run):
    return f"| {label} | " + " | ".join(cell(run[column]) for column in COLUMNS) + " |"


def summary(out):
    lines = ["# MiniMax-H3-Image CUDA benchmark", ""]
    lines += [f"- {line}" for line in (out / "settings").read_text().splitlines()]
    lines += ["", "Selftests (recorded, not a gate):", ""]
    selftests = (out / "selftests").read_text().splitlines() if (out / "selftests").exists() else ["none ran"]
    lines += [f"- {line}" for line in selftests]
    lines += ["", "Run 1 is cold (start to first picture); 2-4 are warm repeats. Energy is average power times wall.", ""]
    lines += ["| run | " + " | ".join(COLUMNS) + " |", "|---|" + "---:|" * len(COLUMNS)]
    sides = {}
    for path in out.glob("*-*.time"):
        side, n = map(int, path.stem.split("-"))
        sides.setdefault(side, {})[n] = measure(out / path.stem)
    for side, runs in sorted(sides.items()):
        lines += [row(f"{side}² {'cold' if n == 1 else f'warm {n - 1}'}", runs[n]) for n in sorted(runs)]
        warm = [runs[n] for n in sorted(runs) if n > 1]
        if warm:
            mean = {}
            for column in COLUMNS:
                values = [run[column] for run in warm if run[column] is not None]
                mean[column] = statistics.fmean(values) if len(values) == len(warm) else None
            lines.append(row(f"{side}² warm mean", mean))
    return "\n".join(lines) + "\n"


def main(argv):
    out = Path(argv[1])
    text = summary(out)
    (out / "SUMMARY.md").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
