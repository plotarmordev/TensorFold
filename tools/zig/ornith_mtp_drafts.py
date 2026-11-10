"""The Ornith-1.5-122B MTP layer's greedy drafts through mlx-lm on tools/bench_prompts.json, for qwen27's mtp_test.zig.

Usage: ornith_mtp_drafts.py MLX_4BIT_DIR BF16_MTP_DIR OUT.safetensors

Each prompt (chat items through the template, thinking off) and mlx-lm's greedy 128-token continuation, fed as the
engine's test feeds it: the prompt in one pass, then the reply a token a step, the target and the MTP layer
(ornith_mtp_reference.mtp_head) each on its own K/V cache. After each target pass the layer reads its last-layer rows
before the final norm beside each row's next token, and the target's head drafts from its last row: one draft a step,
with its lead over the runner-up logit.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache

from ornith_mtp_reference import mtp_head, mtp_rows

STEPS = 128
PROMPTS = Path(__file__).resolve().parents[1] / "bench_prompts.json"


def main(target_dir, mtp_dir, out_path):
    model, tok = load(target_dir)
    lm = model.language_model
    text = lm.model
    prompts = []
    for item in json.loads(PROMPTS.read_text()):
        if item["kind"] == "chat":
            ids = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], add_generation_prompt=True, enable_thinking=False)
        else:
            ids = tok.encode(item["prompt"])
        reply = [int(t) for t, _ in generate_step(mx.array(ids), model, max_tokens=STEPS)]
        prompts.append(list(ids) + reply)
    text.norm = lambda x: x  # the rows the head reads: the last layer's, before the final norm
    head = mtp_head(mtp_dir, lm.args)
    drafts, leads = [], []
    for seq in prompts:
        prompt = len(seq) - STEPS
        cache, mtp_cache = make_prompt_cache(lm), KVCache()
        spans = [(0, prompt)] + [(prompt + k - 1, prompt + k) for k in range(1, STEPS)]
        for first, end in spans:
            hidden = text(mx.array([seq[first:end]]), cache=cache)[0]
            row = mtp_rows(text, head, hidden, mx.array(seq[first + 1 : end + 1]), mtp_cache)[-1]
            logits = lm.lm_head(row).astype(mx.float32)
            top = mx.sort(logits)[-2:]
            drafts.append(mx.argmax(logits).astype(mx.uint32))
            leads.append(top[1] - top[0])
    tensors = {
        "ids": mx.array([t for seq in prompts for t in seq], dtype=mx.uint32),
        "lengths": mx.array([len(seq) for seq in prompts], dtype=mx.uint32),
        "drafts": mx.stack(drafts),
        "leads": mx.stack(leads),
    }
    mx.save_safetensors(out_path, tensors)
    print(len(prompts), "prompts,", len(prompts) * STEPS, "drafts ->", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
