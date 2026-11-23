"""45K prefill tok/s: prompt tokens / time to the first streamed token; a second ask of the same prompt shows the
cache's hit. Three distinct prompts (cold each), then the median."""
import json, statistics, sys, time, urllib.request

base = sys.argv[1]
words = open(sys.argv[2]).read().split()


def ask(text, max_tokens=8):
    body = json.dumps({"model": "m", "stream": True, "max_tokens": max_tokens, "temperature": 0,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None; out = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                if piece:
                    first = first or time.perf_counter()
                    out.append(piece)
    return usage, first - t0, "".join(out)


def tokens(text):
    body = json.dumps({"model": "m", "prompt": text}).encode()
    req = urllib.request.Request(base + "/tokenize", body, {"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req))["tokens"])


rates = []
for salt in range(3):
    text = f"Run {salt}. Summarize the last clause.\n"
    i = salt
    while tokens(text) < 45000:
        text += " ".join(words[i % len(words):][:3000]) + f" [{i}]\n"
        i += 7
    usage, ttft, fresh = ask(text)
    again, ttft2, resumed = ask(text)
    rate = usage["prompt_tokens"] / ttft
    rates.append(rate)
    cached = (again.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"prompt {usage['prompt_tokens']} ttft {ttft:.2f}s prefill {rate:.0f} tok/s; again ttft {ttft2:.2f}s "
          f"cached {cached} same {fresh == resumed}", flush=True)
print(f"median prefill {statistics.median(rates):.0f} tok/s")
