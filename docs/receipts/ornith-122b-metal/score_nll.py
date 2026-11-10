"""Mean per-token NLL, under mlx-lm's teacher-forced forward of the publisher weights, of TensorFold's and mlx-lm's own greedy continuations."""
import json, sys, urllib.request
from mlx_lm import load
import mlx.core as mx

url, model_dir, ref_path, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
model, _ = load(model_dir)

def nll(prompt, cont):
    logits = model(mx.array([prompt + cont]))[0, len(prompt) - 1:-1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -float(mx.mean(mx.take_along_axis(lp, mx.array(cont)[:, None], axis=-1)))

tot_tf = tot_ref = 0.0
rows = json.load(open(ref_path))
for r in rows:
    body = {"model": "bench", "prompt": r["prompt_ids"], "max_tokens": n, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    got = json.load(urllib.request.urlopen(req, timeout=600))["tensorfold"]["token_ids"][:n]
    a, b = nll(r["prompt_ids"], got), nll(r["prompt_ids"], r["ids"][:n])
    tot_tf += a; tot_ref += b
    print(f'{r["name"]}: tensorfold {a:.4f}, mlx-lm {b:.4f}', flush=True)
print(f"mean NLL tensorfold {tot_tf / len(rows):.4f}, mlx-lm {tot_ref / len(rows):.4f}")
