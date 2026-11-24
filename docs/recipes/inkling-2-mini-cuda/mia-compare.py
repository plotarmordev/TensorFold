#!/usr/bin/env python3
"""One arm of card 256's paired run: greedy decode at 1/2/4/8 streams, prefill 4k..128k, and agreement receipts.

    mia-compare.py --url http://127.0.0.1:8188 --out arm.jsonl

Every request is greedy (temperature 0, top_p 1, fixed seed) and thinking stays on the server's default, so the two
arms send identical bodies; the settings that differ (KV dtype, drafts, context) are the server's, not this client's.
Prefill texts are fresh per repetition (a different seed), so no prompt cache is reused: the row reports the cached
tokens the server saw. Decode totals are the sum of the server's per-stream decode rates, median of --reps runs, as
Mia's tools/bench_streams.py reports them. Agreement rows carry the server's token sha and a sha256 of the reply text.
"""
import argparse
import hashlib
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PROSE = "Explain why the sky is blue in a few paragraphs."
CODE = "Write a Python quicksort with docstring and tests."
AGREE = [
    "What is 17 multiplied by 23? Answer with the number only.",
    "Name the three laws of thermodynamics in one sentence each.",
    "Write the first eight Fibonacci numbers, comma separated.",
    "Which ocean is the deepest, and how deep is it?",
    "Summarize what a mixture-of-experts layer does in two sentences.",
]

_NOUNS = ("river mountain lantern harbor meadow orchard village bridge window garden engine letter signal market "
          "teacher farmer sailor painter traveler merchant scholar shepherd weaver miller baker neighbor forest "
          "valley island road kitchen library tower field winter summer morning evening storm cloud candle basket "
          "wagon ladder kettle quilt fence stream pebble feather shadow chapel tavern mill").split()
_ADJS = ("quiet old narrow bright gray patient distant gentle crooked steady golden cold warm early late small "
         "wide heavy simple careful humble faded green silver hollow").split()
_VERBS = ("watched carried followed remembered crossed repaired described opened gathered visited painted mended "
          "counted studied lifted wrapped measured noticed planted guarded").split()
_ENDS = ("before the rain came", "after the long harvest", "while the bells rang", "at the edge of the town",
         "under a pale sky", "beside the slow water", "as the lamps were lit", "for the rest of the season",
         "without a single word", "near the end of the road", "when the frost had gone", "in the usual way")


def prose(words, seed):
    """Deterministic pseudo-prose of `words` words, the same text for the same seed on either arm."""
    rng = random.Random(seed)
    out = []
    while len(out) < words:
        out.append(f"The {rng.choice(_ADJS)} {rng.choice(_NOUNS)} {rng.choice(_VERBS)} the "
                   f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)} {rng.choice(_ENDS)}.")
    return " ".join(out[:words])


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def token_count(base, text):
    """The server's own count of `text` (vLLM-style /tokenize), or None when it has no such route."""
    try:
        reply = post(base, "/tokenize", {"prompt": text, "add_special_tokens": False}, 600)
    except (OSError, ValueError, urllib.error.HTTPError):
        return None
    if isinstance(reply.get("count"), int):
        return reply["count"]
    if isinstance(reply.get("tokens"), list):
        return len(reply["tokens"])
    return None


def sized_prose(base, want, seed):
    """About `want` tokens of prose: scale the word count by the server's own count, then keep the words that fit."""
    words = max(8, int(want * 0.8))
    text = prose(words, seed)
    got = token_count(base, text)
    if got:
        words = max(8, int(words * want / got))
        text = prose(words, seed)
    return text


def chat(base, model, prompt, max_tokens, timeout):
    """One streamed greedy request; a receipt dict, never an exception (HTTP and transport failures are recorded)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "seed": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    row = {"wall_s": None, "ttft_s": None, "prompt_tokens": None, "cached_tokens": None,
           "completion_tokens": None, "prefill_s": None, "decode_tps": None, "token_sha": None,
           "finish": None, "failed": 0, "errors": []}
    text = []
    usage, extras = {}, {}
    t0 = time.time()
    try:
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise ValueError(json.dumps(chunk["error"])[:300])
                for key, slot in (("usage", usage), ("tensorfold", extras)):
                    if isinstance(chunk.get(key), dict):
                        slot.update(chunk[key])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece:
                        text.append(piece)
                        if row["ttft_s"] is None:
                            row["ttft_s"] = round(time.time() - t0, 4)
                    if choice.get("finish_reason"):
                        row["finish"] = choice["finish_reason"]
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        row["failed"] = 1
        row["errors"].append(f"{type(exc).__name__}: {exc}"[:300])
    row["wall_s"] = round(time.time() - t0, 4)
    row["prompt_tokens"] = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details") or {}
    row["cached_tokens"] = details.get("cached_tokens", usage.get("cached_tokens"))
    row["completion_tokens"] = usage.get("completion_tokens")
    row["prefill_s"] = extras.get("prefill_seconds", extras.get("prefill_s"))
    row["decode_tps"] = extras.get("tokens_per_second")
    row["token_sha"] = extras.get("token_sha")
    joined = "".join(text)
    row["text_sha256"] = hashlib.sha256(joined.encode()).hexdigest()[:16]
    if row["ttft_s"] is None and extras.get("time_to_first_token") is not None:
        row["ttft_s"] = round(float(extras["time_to_first_token"]), 4)
    return row


def emit(out, row):
    out.write(json.dumps(row) + "\n")
    out.flush()


def decode(base, model, out, streams, tokens, reps, timeout):
    """Total and per-stream decode at each level, median of `reps` runs, for the prose and code prompts."""
    for name, prompt in (("prose", PROSE), ("code", CODE)):
        chat(base, model, prompt, 16, timeout)  # warm up
        for level in streams:
            runs = []
            for _ in range(reps):
                rows = [None] * level

                def one(i):
                    rows[i] = chat(base, model, prompt, tokens, timeout)

                threads = [threading.Thread(target=one, args=(i,)) for i in range(level)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                failed = sum(r["failed"] for r in rows)
                per = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
                row = {"kind": "decode", "prompt": name, "streams": level, "rep": _,
                       "aggregate_tps": round(sum(per), 2) if per else None,
                       "per_stream_tps": round(statistics.mean(per), 2) if per else None,
                       "ttft_s": round(statistics.median(r["ttft_s"] for r in rows), 4),
                       "wall_s": round(max(r["wall_s"] for r in rows), 3),
                       "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
                       "failed": failed,
                       "errors": [e for r in rows for e in r["errors"]][:5]}
                runs.append(row)
                emit(out, row)
            ok = [r for r in runs if r["aggregate_tps"] is not None]
            if ok:
                print(f"{name} {level} stream(s): total {statistics.median(r['aggregate_tps'] for r in ok):.1f} tok/s,"
                      f" per stream {statistics.median(r['per_stream_tps'] for r in ok):.1f},"
                      f" TTFT {statistics.median(r['ttft_s'] for r in ok) * 1000:.0f} ms,"
                      f" failed {sum(r['failed'] for r in runs)} (median of {len(ok)})", flush=True)


def prefill(base, model, out, sizes, reps, timeout):
    """One fresh prompt per repetition at each size; the row reports the server's prefill seconds and cached tokens."""
    for want in sizes:
        runs = []
        for k in range(reps):
            prompt = sized_prose(base, want, seed=want + 7919 * k)
            row = chat(base, model, prompt, 16, timeout)
            row.update({"kind": "prefill", "want_tokens": want, "rep": k})
            runs.append(row)
            emit(out, row)
        ok = [r for r in runs if not r["failed"] and r["prefill_s"]]
        if ok:
            best = min(ok, key=lambda r: r["prefill_s"])
            fresh = best["prompt_tokens"] - (best["cached_tokens"] or 0)
            print(f"prefill ~{want}: {best['prompt_tokens']} tok, {fresh} prefilled in {best['prefill_s']:.2f} s,"
                  f" {fresh / best['prefill_s']:.0f} tok/s, TTFT {best['ttft_s']:.2f} s,"
                  f" cached {best['cached_tokens']}, failed {sum(r['failed'] for r in runs)}", flush=True)


def agree(base, model, out, timeout):
    """The same greedy prompts on both arms: token sha and text sha are what the shell compares."""
    prompts = [(i, p) for i, p in enumerate(AGREE)]
    prompts.append((len(AGREE), sized_prose(base, 4096, seed=4096) + "\n\nSummarize the text above in one sentence."))
    for idx, prompt in prompts:
        row = chat(base, model, prompt, 96, timeout)
        row.update({"kind": "agree", "prompt_idx": idx})
        emit(out, row)
        print(f"agree {idx}: ok {not row['failed']}, tokens {row['completion_tokens']},"
              f" token sha {row['token_sha']}, text {row['text_sha256']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="inkling-2-mini")
    p.add_argument("--streams", default="1,2,4,8")
    p.add_argument("--sizes", default="4096,8192,16384,32768,65536,131000")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--phase", choices=("decode", "prefill", "agree"), default=None,
                   help="one phase, so the job can sample memory between them (default: all three)")
    a = p.parse_args()
    base = a.url.rstrip("/")
    with open(a.out, "a") as out:
        if a.phase in (None, "decode"):
            decode(base, a.model, out, [int(s) for s in a.streams.split(",")], a.tokens, a.reps, a.timeout)
        if a.phase in (None, "prefill"):
            prefill(base, a.model, out, [int(s) for s in a.sizes.split(",")], a.reps, a.timeout)
        if a.phase in (None, "agree"):
            agree(base, a.model, out, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
