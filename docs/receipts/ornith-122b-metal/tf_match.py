"""Send mlx_ref.py's prompt ids to a TensorFold server greedily; count leading token matches against mlx-lm's."""
import json, sys, urllib.request

url, ref_path, n, draft = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "draft"
rows = []
for r in json.load(open(ref_path)):
    body = {"model": "bench", "prompt": r["prompt_ids"], "max_tokens": n, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True, "draft": draft}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    got = json.load(urllib.request.urlopen(req, timeout=600))["tensorfold"]["token_ids"]
    want = r["ids"][:n]
    lead = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    same = sum(a == b for a, b in zip(got, want))
    rows.append({"name": r["name"], "lead": lead, "same": same, "n": len(want)})
    print(f'{r["name"]}: first {lead}/{len(want)} match, {same} equal positions', flush=True)
print("leading-match total", sum(x["lead"] for x in rows), "/", sum(x["n"] for x in rows))
