"""Recompute the recipe's 12 November speed, acceptance, token-equality and NLL figures from the receipts here."""
import hashlib, json, statistics, struct
from pathlib import Path

here = Path(__file__).parent
load = lambda name: json.loads((here / name).read_text())

def u32_sha(rows, key):
    return hashlib.sha256(b"".join(struct.pack("<I", t) for r in rows for t in r[key])).hexdigest()

def ratio(accepted, drafted):
    return f"{100 * accepted / drafted:.1f}% ({accepted}/{drafted})" if drafted else f"- ({accepted}/{drafted})"

# Speed: each server's raw runs end with its timed replies (after the warmup and any equality controls)
print("mode   par streams tok/s/stream aggregate ttft median/max accepted/proposed")
for mode in ("plain", "dflash", "mtp"):
    for par in (1, 4):
        doc, raw = load(f"bench-{mode}-p{par}-fresh.json"), load(f"bench-{mode}-p{par}-fresh.raw.json")
        timed = raw["runs"][-(3 if par == 1 else 15):]
        for i, cell in enumerate(doc["cells"]):
            assert cell["failed"] == cell["unmeasured"] == 0 and cell["timing_valid"]
            if par == 4:
                assert cell["alone"]["unequal"] == cell["alone"]["failed"] == 0
                if mode != "plain":
                    assert cell["serial"] == {"equal": 4, "unequal": 0, "failed": 0}
            runs = timed if par == 1 else timed[:3] if i == 0 else timed[3:]
            assert len(runs) == 3 * cell["streams"]
            spec = [r["speculative"] for r in runs]
            print(f'{mode:6} {par}   {cell["streams"]}       {cell["per_stream_tps"]:6.1f}       {cell["aggregate_tps"]:6.1f}    '
                  f'{cell["request"]["ttft_s"]:.3f}/{max(r["ttft_s"] for r in runs):.3f}  '
                  f'{ratio(sum(s["accepted"] for s in spec), sum(s["drafted"] for s in spec))}  '
                  f'reps {[round(r["aggregate_tps"], 1) for r in cell["reps"]]}')

# MTP on a server restarted before each one-stream request, against plain's reply to the same prompt
plain_sha = load("bench-plain-p1-fresh.raw.json")["runs"][-1]["token_sha256"]
restart = [load(f"bench-mtp-p1-restart-{i}.raw.json")["runs"][-1] for i in range(3)]
assert all(r["token_sha256"] == plain_sha for r in restart)
print("MTP restarted: tok/s", [round(r["decode_tps"], 1) for r in restart],
      f'median {statistics.median(r["decode_tps"] for r in restart):.1f}, '
      f'ttft median/max {statistics.median(r["ttft_s"] for r in restart):.3f}/{max(r["ttft_s"] for r in restart):.3f}, '
      + ratio(sum(r["speculative"]["accepted"] for r in restart), sum(r["speculative"]["drafted"] for r in restart))
      + ", tokens equal plain")

# Quality: 16 prompts x 128 greedy tokens; MTP ran as four concurrent groups of four
plain, dflash, mtp = load("quality-plain.json"), load("quality-dflash.json"), load("quality-mtp-shared.json")
assert [r["ids"] for r in plain] == [r["ids"] for r in dflash] == [r["ids"] for r in mtp]
assert all(r["speculative"]["drafted"] > 0 for r in mtp)
print("quality: plain, DFlash2 and MTP ids equal,", sum(len(r["ids"]) for r in plain), "tokens; MTP",
      ratio(sum(r["speculative"]["accepted"] for r in mtp), sum(r["speculative"]["drafted"] for r in mtp)))
print("leading match against mlx-lm", sum(r["lead"] for r in plain), "/", sum(len(r["reference_ids"]) for r in plain))
print("native ids sha256", u32_sha(plain, "ids"))
print("mlx-lm ids sha256", u32_sha(plain, "reference_ids"))
print("prompt fixture sha256", hashlib.sha256((here / "../../../tools/bench_prompts.json").read_bytes()).hexdigest())

nll = load("score-nll.json")
assert [r["name"] for r in nll] == [r["name"] for r in plain]
print(f'mean NLL tensorfold {sum(r["tensorfold"] for r in nll) / len(nll):.7f}, '
      f'mlx-lm {sum(r["mlx_lm"] for r in nll) / len(nll):.7f}')
