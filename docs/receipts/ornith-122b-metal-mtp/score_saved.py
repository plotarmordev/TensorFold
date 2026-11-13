"""Mean per-token NLL, under mlx-lm's teacher-forced forward of the publisher weights, of the saved native and
mlx-lm continuations in quality-plain.json; writes one row per prompt as JSON."""
import json, sys
from mlx_lm import load
import mlx.core as mx

model_dir, quality_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
model, _ = load(model_dir)

def nll(prompt, cont):
    logits = model(mx.array([prompt + cont]))[0, len(prompt) - 1:-1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -float(mx.mean(mx.take_along_axis(lp, mx.array(cont)[:, None], axis=-1)))

rows = []
for r in json.load(open(quality_path)):
    rows.append({"name": r["name"], "tensorfold": nll(r["prompt_ids"], r["ids"]), "mlx_lm": nll(r["prompt_ids"], r["reference_ids"])})
    print(f'{r["name"]}: tensorfold {rows[-1]["tensorfold"]:.4f}, mlx-lm {rows[-1]["mlx_lm"]:.4f}', flush=True)
json.dump(rows, open(out_path, "w"), indent=1)
print(f'mean NLL tensorfold {sum(r["tensorfold"] for r in rows) / len(rows):.7f}, '
      f'mlx-lm {sum(r["mlx_lm"] for r in rows) / len(rows):.7f}')
