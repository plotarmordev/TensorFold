#!/usr/bin/env python3
"""Bursts of identical requests through the recipe's bench.py: identical prompts in waves of N at once.

  API_URL=... python3 waves.py OUT.jsonl [CASE:N[:distinct] ...]   default: count:4 count:8 prose:4 prose:8 code:4 code:8

Each case: one reply alone (the reference), then a warmup wave and 3 measured waves of N identical requests released
together; 400 output tokens each (ignore_eos), greedy, thinking off, streamed. `distinct` gives stream k its own prompt
("Stream k: " before the same task, the same length for k up to 9) and its own reply alone, so the streams never run
in lockstep on the same tokens. Per wave it records the wave aggregate (output tokens over dispatch to last token,
prefill included), bench.py's decode aggregate (first to last text), each lane's time to first text from dispatch and
the server's TTFT, and which replies equal their reply alone. Every wave goes to OUT.jsonl; a summary line per case to
stdout.
"""
import json
import statistics
import sys
import threading
import time

import bench
import prompts

WAVES = 3
TOKENS = 400


def wave(bodies):
    bench.cool()
    n = len(bodies)
    gate = threading.Barrier(n + 1)
    out = [None] * n

    def run(i):
        try:
            out[i] = bench.generate("/v1/chat/completions", bodies[i], gate)
        except Exception as e:  # re-raised below, on the main thread
            gate.abort()
            out[i] = e

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    gate.wait(timeout=60)
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    if errors := [r for r in out if isinstance(r, Exception)]:
        raise errors[0]
    return t0, out


def stream_prompt(prompt, k):
    """The prompt with "Stream k: " before its (single, text) user message."""
    (message,) = prompt["messages"]
    return {**prompt, "messages": [{**message, "content": f"Stream {k}: {message['content']}"}]}


def main():
    path, cases = sys.argv[1], sys.argv[2:] or ["count:4", "count:8", "prose:4", "prose:8", "code:4", "code:8"]
    chosen = prompts.prompts()
    with open(path, "a") as log:
        for case in cases:
            name, n, *mode = case.split(":")
            n, distinct = int(n), mode == ["distinct"]
            if distinct:
                bodies = [{**bench.chat(stream_prompt(chosen[name], k + 1)), "max_tokens": TOKENS} for k in range(n)]
            else:
                bodies = [{**bench.chat(chosen[name]), "max_tokens": TOKENS}] * n
            alones = []
            for body in bodies[:n if distinct else 1]:
                bench.cool()
                alones.append(bench.generate("/v1/chat/completions", body))
            alone = alones * (1 if distinct else n)
            label = f"{name}-distinct" if distinct else name
            rows = []
            for w in range(WAVES + 1):
                t0, runs = wave(bodies)
                end = max(r["last"] for r in runs)
                row = {
                    "case": label, "n": n, "alones": len(alones), "wave": w, "warmup": w == 0,
                    "wave_tps": sum(r["tokens"] for r in runs) / (end - t0),
                    "decode_tps": sum(r["tokens"] - 1 for r in runs) / (end - min(r["first"] for r in runs)),
                    "first_ms": sorted(round((r["first"] - t0) * 1e3, 1) for r in runs),
                    "server_ttft_ms": sorted(round(r["ttft"] * 1e3, 1) for r in runs),
                    "wall_s": round(end - t0, 3),
                    "tokens": [r["tokens"] for r in runs],
                    "equal_alone": sum(r["sha256"] == a["sha256"] for r, a in zip(runs, alone)),
                    "alone_sha256": alone[0]["sha256"], "alone_tps": alone[0]["tps"],
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if w:
                    rows.append(row)
            print(f"{label:6} c{n}  wave  {statistics.median(r['wave_tps'] for r in rows):7.2f} tok/s  "
                  f"decode {statistics.median(r['decode_tps'] for r in rows):7.2f} tok/s  "
                  f"first-text median {statistics.median(statistics.median(r['first_ms']) for r in rows):7.1f} ms  "
                  f"worst {max(max(r['first_ms']) for r in rows):7.1f} ms  "
                  f"{sum(r['equal_alone'] for r in rows)}/{WAVES * n} equal alone  "
                  f"alone {alone[0]['tps']:.2f} tok/s {alone[0]['sha256'][:12]}", flush=True)


if __name__ == "__main__":
    bench.request("/health").close()  # reach the server before any wait for cooling
    main()
