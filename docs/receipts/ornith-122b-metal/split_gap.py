"""At each prompt's first TensorFold/mlx-lm split, mlx-lm's logit gap between its token and TensorFold's (bf16 near-tie check)."""
import json, sys, urllib.request
from mlx_lm import load
import mlx.core as mx

url, model_dir, ref_path, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
model, _ = load(model_dir)
for r in json.load(open(ref_path)):
    body = {"model": "bench", "prompt": r["prompt_ids"], "max_tokens": n, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    got = json.load(urllib.request.urlopen(req, timeout=600))["tensorfold"]["token_ids"]
    want = r["ids"][:n]
    i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if i is None:
        print(f'{r["name"]}: no split in {n}'); continue
    logits = model(mx.array([r["prompt_ids"] + want[:i]]))[0, -1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits)
    print(f'{r["name"]}: split at {i}, mlx {want[i]} vs tf {got[i]}, logprob gap {float(lp[want[i]] - lp[got[i]]):.4f}', flush=True)
