"""The Ornith-1.5-122B MTP layer's reference rows through mlx-lm, for qwen27's mtp_test.zig.

Usage: ornith_mtp_reference.py MLX_4BIT_DIR BF16_MTP_DIR OUT.safetensors

The target's last-layer rows before its final norm over a fixed prompt, then the bf16 mtp.* weights quantized as
the engine loads them (affine 4-bit, group 64, every matrix) and the layer's forward: fc over
[norm_e(embed(token j + 1)) | norm_h(hidden j)], one full-attention MoE block at positions 0..n-1, mtp.norm.
"""
import glob
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.qwen3_5 import DecoderLayer

PROMPT = "def fibonacci(n):\n    if n < 2:"
NORMS = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm", "pre_fc_norm", "mtp.norm")


def mtp_weights(folder, experts):
    w = {}
    for path in sorted(glob.glob(f"{folder}/*.safetensors")):
        w.update({k[len("mtp."):]: v for k, v in mx.load(path).items() if k.startswith("mtp.")})
    # Qwen's RMSNorm scales by 1 + w; mlx-lm's by w
    w = {k: v + 1 if any(n in "mtp." + k for n in NORMS) else v for k, v in w.items()}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        w[f"layers.0.mlp.switch_mlp.{proj}.weight"] = mx.stack([w.pop(f"layers.0.mlp.experts.{e}.{proj}.weight") for e in range(experts)])
    return w


def mtp_head(mtp_dir, args):
    """The MTP layer as the engine loads it: (fc, block, norm), every matrix affine 4-bit g64; norm(x, name) gains `name`."""
    w = mtp_weights(mtp_dir, args.num_experts)
    layer = DecoderLayer(args, args.full_attention_interval - 1)
    layer.load_weights([(k[len("layers.0."):], v) for k, v in w.items() if k.startswith("layers.0.")])
    fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
    fc.load_weights([("weight", w["fc.weight"])])
    nn.quantize(layer, group_size=64, bits=4)
    fc = nn.QuantizedLinear.from_linear(fc, group_size=64, bits=4)  # nn.quantize skips a bare root module
    norm = lambda x, name: mx.fast.rms_norm(x, w[f"{name}.weight"], args.rms_norm_eps)
    return fc, layer, norm


def mtp_rows(text, head, hidden, nxt, cache=None):
    """The layer's normed rows over one causal pass: row j reads hidden j beside token nxt[j]; `cache` its own K/V."""
    fc, layer, norm = head
    x = mx.concatenate([norm(text.embed_tokens(nxt), "pre_fc_norm_embedding"), norm(hidden, "pre_fc_norm_hidden")], axis=-1)
    return norm(layer(fc(x)[None], mask="causal" if len(nxt) > 1 else None, cache=cache)[0], "norm")


def main(target_dir, mtp_dir, out_path):
    model, tok = load(target_dir)
    text = model.language_model.model
    args = model.language_model.args
    ids = tok.encode(PROMPT)
    text.norm = lambda x: x  # the rows the head reads: the last layer's, before the final norm
    hidden = text(mx.array([ids]))[0]
    nxt = mx.array(ids[1:])
    out = mtp_rows(text, mtp_head(mtp_dir, args), hidden[:-1], nxt)
    tensors = {"ids": nxt.astype(mx.uint32), "hidden": hidden[:-1].astype(mx.bfloat16), "out": out.astype(mx.bfloat16)}
    mx.save_safetensors(out_path, tensors)
    print(len(ids) - 1, "rows ->", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])
