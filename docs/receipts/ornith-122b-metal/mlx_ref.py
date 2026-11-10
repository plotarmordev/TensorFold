"""Greedy token ids from mlx-lm for tools/bench_prompts.json (chat items through the template, thinking off)."""
import json, sys
from mlx_lm import load
from mlx_lm.generate import generate_step
import mlx.core as mx

model_dir, prompts_path, out_path, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
model, tok = load(model_dir)
out = []
for item in json.load(open(prompts_path)):
    if item["kind"] == "chat":
        ids = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], add_generation_prompt=True, enable_thinking=False)
    else:
        ids = tok.encode(item["prompt"])
    got = []
    for t, _ in generate_step(mx.array(ids), model, max_tokens=n):
        got.append(int(t))
    out.append({"name": item["name"], "prompt_ids": list(ids), "ids": got})
    print(item["name"], len(ids), got[:8], flush=True)
json.dump(out, open(out_path, "w"))
