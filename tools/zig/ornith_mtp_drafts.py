"""The Ornith-1.5-122B MTP layer's greedy drafts through mlx-lm on tools/bench_prompts.json, for qwen27's mtp_test.zig.

Usage: ornith_mtp_drafts.py MLX_4BIT_DIR BF16_MTP_DIR OUT.safetensors

Each prompt (chat items through the template, thinking off) and mlx-lm's greedy 128-token continuation; over them the
target's last-layer rows before its final norm, then the MTP layer (ornith_mtp_reference.mtp_head) in one causal pass
and the target's head. Row j reads hidden j beside token j + 1 and drafts token j + 2: the drafts from the prompt's
last row on, one a decode step, with each draft's lead over the runner-up logit.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

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
        hidden = text(mx.array([seq]))[0]
        rows = mtp_rows(text, head, hidden[:-1], mx.array(seq[1:]))[len(seq) - STEPS - 1 :]
        logits = lm.lm_head(rows).astype(mx.float32)
        top = mx.sort(logits, axis=-1)[:, -2:]
        drafts.append(mx.argmax(logits, axis=-1).astype(mx.uint32))
        leads.append(top[:, 1] - top[:, 0])
    tensors = {
        "ids": mx.array([t for seq in prompts for t in seq], dtype=mx.uint32),
        "lengths": mx.array([len(seq) for seq in prompts], dtype=mx.uint32),
        "drafts": mx.concatenate(drafts),
        "leads": mx.concatenate(leads),
    }
    mx.save_safetensors(out_path, tensors)
    print(len(prompts), "prompts,", len(prompts) * STEPS, "drafts ->", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
