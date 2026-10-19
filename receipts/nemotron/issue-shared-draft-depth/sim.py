#!/usr/bin/env python3
"""Replay the lanes core's shared-round depth choice on a hold's own cost table: Rule.budgets (allocateDrafts with
batched head steps), windows.allocate's trim of the held drafts, and observeDepth (core/lanes/depth.zig, allocate.zig,
windows.zig at v1.1.0, 76aafcf2). The learned shared overhead is OVERHEAD ms (default 0: observeOverhead clamps a
round's ms past the table's model at 0, and the measured c4 rounds ran under that model, judged from wall ms a round).

Usage: sim.py TABLE.bin STREAMS ACCEPT MODE [TOKENS] [SEEDS] [OVERHEAD]
  ACCEPT  chance each draft lands given the one before landed (count: 1.0, every draft landed in every hold)
  MODE    none: 1.1.0; every8: one draft past the grant every 8th round (the lone rule's probe);
          landed: one draft past the grant after a round that kept a stream's whole chain;
          frontier: a depth no round verified is priced no lower than the deepest verified one (explore.patch);
          ratio: as frontier, scaled by the prior's own step between the two depths
Prints rounds a request, drafts a stream a round and the table's model of a request's ms."""
import random
import struct
import sys

PRIOR = [0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5]  # families/nemotron/config.zig draft_prior
DEPTH_RATE = 0.15  # lanes/depth.zig Constants.depth_rate
BATCH_ROWS = 32  # cuda_lanes facts: batch_rows = windowRows


def load(path):
    raw = open(path, "rb").read()
    assert raw[:6] == b"TFWC2\n", path
    *verify, rows, level = struct.unpack("<33dQd", raw[6:])
    return verify[: rows + 1], level


def chain(rates, n):
    out, reach = [], 1.0
    for j in range(n):
        reach *= rates[j] if j < len(rates) else rates[-1]
        out.append(reach)
    return out


def allocate(probs, cost, max_rows, step_ms, overhead):
    """allocateDrafts, HeadCost{batched}: one fixed row a stream, a head step per level of the deepest stream."""
    counts = [0] * len(probs)
    best_counts = counts[:]
    rows = len(probs)
    expected = float(rows)
    deepest = 0
    best = expected / (cost[rows] + overhead)
    while rows < max_rows:
        pick, pick_p = None, 0.0
        for s, p in enumerate(probs):
            if counts[s] < len(p) and (pick is None or p[counts[s]] > pick_p):
                pick, pick_p = s, p[counts[s]]
        if pick is None:
            break
        counts[pick] += 1
        rows += 1
        deepest = max(deepest, counts[pick])
        expected += pick_p
        now = expected / (cost[rows] + overhead + step_ms * deepest)
        if now > best:
            best, best_counts = now, counts[:]
    return best_counts


def run(cost, level, streams, accept, probe, tokens, rng, overhead):
    rates = [PRIOR[:] for _ in range(streams)]
    made = [1] * streams  # the prompt's pending token
    held = [1] * streams  # drafts the head made for the next round
    landed = [False] * streams
    verified = [0] * streams  # depths a round has verified
    rounds = [0] * streams
    drafted = 0
    ms = 0.0
    most = BATCH_ROWS - 1
    while True:
        live = [s for s in range(streams) if made[s] < tokens]  # a finished stream leaves the shared round
        if not live:
            break
        trimmed = allocate([chain(rates[s], min(held[s], tokens - made[s] - 1)) for s in live], cost, BATCH_ROWS, 0.0,
                           overhead)
        ms += cost[len(live) + sum(trimmed)] + overhead + level * max(held[s] for s in live)
        for s, d in zip(live, trimmed):
            rounds[s] += 1
            kept = 0
            while kept < d and rng.random() < accept:
                kept += 1
            for j in range(min(d, len(rates[s]))):
                if kept < j:
                    break
                rates[s][j] += DEPTH_RATE * ((1.0 if kept > j else 0.0) - rates[s][j])
            landed[s] = d > 0 and kept == d
            verified[s] = max(verified[s], min(d, kept + 1, len(rates[s])))
            f = verified[s]
            if probe == "frontier" and 0 < f < len(rates[s]):
                rates[s][f] = max(rates[s][f], rates[s][f - 1])
            if probe == "ratio" and 0 < f < len(rates[s]):
                rates[s][f] = max(rates[s][f], rates[s][f - 1] * PRIOR[f] / PRIOR[f - 1])
            made[s] += 1 + kept
            drafted += d
        granted = allocate([chain(rates[s], most) for s in live], cost, BATCH_ROWS, level, overhead)
        for s, c in zip(live, granted):
            held[s] = max(1, c)
            if (probe == "every8" and rounds[s] % 8 == 0) or (probe == "landed" and landed[s]):
                held[s] = min(most, held[s] + 1)
    return sum(rounds) / streams, drafted / sum(rounds), ms


def main():
    cost, level = load(sys.argv[1])
    streams, accept, probe = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    seeds = int(sys.argv[6]) if len(sys.argv) > 6 else (1 if accept == 1.0 else 20)
    overhead = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    runs = [run(cost, level, streams, accept, probe, tokens, random.Random(seed), overhead) for seed in range(seeds)]
    rounds, per, ms = (sum(r[i] for r in runs) / seeds for i in range(3))
    print(f"c{streams} accept {accept} {probe} overhead {overhead:g}: {rounds:.1f} rounds a request, "
          f"{per:.2f} drafts a stream a round, model {ms / 1000:.2f} s a request")


if __name__ == "__main__":
    main()
