"""Inkling 2 Mini's no-draft decode limit on a bandwidth: python3 ceiling.py MODEL GBPS [STREAMS...].

A round of N streams reads every non-expert weight once (the core layers once a pass, the unembedding whole, the
embedding only a row each, left out) and, a visit, the experts any of its N rows routes to: under uniform routing a
share 1 - (1 - k/E)^N of them. Total tok/s is N rows over that read time, with no drafts and no KV. One stream's
figure is a hard ceiling; past one, real routing overlaps more than uniform, so the figure is an estimate, not a bound.
"""
import glob
import json
import re
import struct
import sys


def round_bytes(model: str, streams: int) -> float:
    text = json.load(open(f"{model}/config.json"))
    text = text.get("text_config", text)
    loop = text["loop_config"]
    touched = 1 - (1 - text["num_experts_per_tok"] / text["n_routed_experts"]) ** streams
    total = 0.0
    for path in glob.glob(f"{model}/*.safetensors"):
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, t in header.items():
            if name == "__metadata__" or not name.startswith("model.llm.") or name == "model.llm.embed.weight":
                continue
            size = t["data_offsets"][1] - t["data_offsets"][0]
            layer = re.match(r"model\.llm\.layers\.(\d+)\.", name)
            if layer:
                if loop["core_start"] <= int(layer[1]) < loop["core_end"]:
                    size *= loop["num_passes"]
                if ".mlp.experts." in name:
                    size *= touched
            total += size
    return total


if __name__ == "__main__":
    model, gbps = sys.argv[1], float(sys.argv[2])
    for n in map(int, sys.argv[3:] or ["1"]):
        b = round_bytes(model, n)
        print(f"RESULT inkling2 mini no-draft limit {n} stream(s): {b / 1e9:.2f} GB a round, {n * gbps * 1e9 / b:.1f} tok/s total at {gbps:.0f} GB/s")
