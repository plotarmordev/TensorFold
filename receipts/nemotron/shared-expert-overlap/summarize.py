"""The receipt tables from qualify.py's runs, in run order. Usage: summarize.py RECEIPTS_DIR LABEL..."""
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
labels = sys.argv[2:]


def load(label, suffix):
    return json.loads((d / f'{label}.{suffix}').read_text())


print('| run | commit | ' + ' | '.join(f'{p} token_sha256 (drafted == draft:false)' for p in ('code', 'chat')) + ' |')
print('|---|---|---|---|')
for label in labels:
    m = load(label, 'json')
    shas = {c['prompt']: c['token_sha256'][:16] for c in m['checks']}
    print(f"| {label} | {m.get('source_commit', m.get('commit'))} | {shas['code']} | {shas['chat']} |")
print()
print('| run | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |')
print('|---|---|---|')
for label in labels:
    rows = load(label, 'bench_openai.json')
    decode = '; '.join(f"{r['prompt']} {r['decode_tps_median']:.2f} ({', '.join(str(x) for x in r['decode_tps_all'])})" for r in rows)
    cells = load(label, 'bench_concurrent.json')['cells']
    alone = sum(c['alone']['equal'] for c in cells), sum(c['alone']['unequal'] for c in cells)
    serial = sum(c['serial']['equal'] for c in cells), sum(c['serial']['unequal'] for c in cells)
    print(f'| {label} | {decode} | {alone[0]} ({alone[1]}) / {serial[0]} ({serial[1]}) |')
print()
print('| run | bench_concurrent aggregate tok/s: code x1, code x4, chat x1, chat x4 | bench_cache cold ttft_s (cached tokens, prompt tokens) | qualified |')
print('|---|---|---|---|')
for label in labels:
    cells = load(label, 'bench_concurrent.json')['cells']
    agg = ', '.join(f"{c['aggregate_tps']}" for c in cells)
    cache = load(label, 'bench_cache.json')
    first = cache['rows'][0]
    print(f"| {label} | {agg} | {first['ttft_s']:.3f} ({first['cached_tokens']}, {first['prompt_tokens']}) | {cache['qualified']} |")
print()
for prompt in ('fibonacci-raw', 'gpu-chat-no-think'):
    for kind in ('base', 'pr'):
        runs = [x for label in labels if label.startswith(kind) for r in load(label, 'bench_openai.json') if r['prompt'] == prompt for x in r['decode_tps_all']]
        print(f'{prompt} {kind}: median of {len(runs)} runs {statistics.median(runs):.2f} tok/s')
