"""The receipt tables from hold.py's runs, in run order. Usage: summarize.py RECEIPTS_DIR LABEL..."""
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
labels = sys.argv[2:]


def load(label, suffix='json'):
    return json.loads((d / f'{label}.{suffix}').read_text())


def heat_waits(label, prompt_tokens):
    """The heat_wait_s the server logged inside each cold request of prompt_tokens, in request order."""
    waits, pending = [], 0.0
    for line in (d / f'{label}.server.log').read_text().splitlines():
        if line.startswith('info: heat_wait_s '):
            pending += float(line.split()[-1])
        elif ' done ' in line:
            if f'prompt={prompt_tokens} cached=0 ' in line:
                waits.append(pending)
            pending = 0.0
    return waits


print('| run | commit | server sha256 | fresh window costs | ready s | heat waits s | admission |')
print('|---|---|---|---|---|---|---|')
for label in labels:
    m = load(label)
    print(f"| {label} | {m['commit']} | {m['binary_sha256']} | {m['costs_cache_deleted_before_boot']} | {m['ready_s']} | {m.get('heat_wait_s')} | {m['admission']} |")
print()
decode = [x for x in labels if load(x)['mode'] == 'decode']
if decode:
    print('| run | code / chat token_sha256 (drafted == draft:false) | bench_openai decode tok/s, median of 3 (all) | bench_concurrent alone / serial equal (unequal) |')
    print('|---|---|---|---|')
    for label in decode:
        shas = ' / '.join(c['token_sha256'][:16] for c in load(label)['checks'])
        rows = load(label, 'bench_openai.json')
        tps = '; '.join(f"{r['prompt']} {r['decode_tps_median']:.2f} ({', '.join(str(x) for x in r['decode_tps_all'])})" for r in rows)
        cells = load(label, 'bench_concurrent.json')['cells']
        alone = sum(c['alone']['equal'] for c in cells), sum(c['alone']['unequal'] for c in cells)
        serial = sum(c['serial']['equal'] for c in cells), sum(c['serial']['unequal'] for c in cells)
        agg = ', '.join(str(c['aggregate_tps']) for c in cells)
        print(f'| {label} | {shas} | {tps}; aggregate {agg} | {alone[0]} ({alone[1]}) / {serial[0]} ({serial[1]}) |')
    print()
prefill = [x for x in labels if load(x)['mode'] != 'decode']
if prefill:
    print('| run | prompt tokens | cold ttft_s: measured - logged heat wait = net | cached tokens | drafted token_sha256 (== draft:false) | qualified |')
    print('|---|---|---|---|---|---|')
    net = {}
    for label in prefill:
        for p in load(label)['prompts']:
            size = p['prompt_tokens']
            rows = load(label, f'bench_cache-{size}.json')['rows']
            waits = heat_waits(label, size)
            cells = [(r['ttft_s'], w) for r, w in zip(rows, waits)]
            net.setdefault((size, label.split('-')[0]), []).extend(t - w for t, w in cells)
            print(f"| {label} | {size} | {', '.join(f'{t:.3f} - {w:.1f} = {t - w:.3f}' for t, w in cells)} | {p['cached_tokens']} | "
                  f"{(p['drafted_token_sha256'] or '')[:16]} ({p['drafted_equals_plain']}) | {p['qualified']} |")
    print()
    print('| prompt tokens | base net ttft_s, median (n) | PR net ttft_s, median (n) | PR / base | token_sha256 equal across runs |')
    print('|---|---|---|---|---|')
    for size in sorted({s for s, _ in net}):
        b, p = net[(size, 'base')], net[(size, 'pr')]
        shas = {x['drafted_token_sha256'] for label in prefill for x in load(label)['prompts'] if x['prompt_tokens'] == size}
        print(f'| {size} | {statistics.median(b):.3f} ({len(b)}) | {statistics.median(p):.3f} ({len(p)}) | {statistics.median(p) / statistics.median(b):.3f} | {len(shas) == 1} |')
