import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
labels = ['base-a', 'pr-a', 'pr-b', 'base-b', 'cap0', 'cap1', 'confidence1']
reference = json.loads((root / 'base-a.json').read_text())
expected = {row['prompt']: row['plain']['token_sha256'] for row in reference['checks']}
summary = {'runs': [], 'passed': 0, 'failed': 0}
for label in labels:
    report = json.loads((root / (label + '.json')).read_text())
    assert report['result'] == 'completed' and report['gpu_idle_after'] and not report['violations']
    run = {'label': label, 'checks': []}
    for row in report['checks']:
        equal = row['drafted']['token_sha256'] == row['plain']['token_sha256'] == expected[row['prompt']]
        summary['passed' if equal else 'failed'] += 1
        run['checks'].append({'prompt': row['prompt'], 'equal_base_plain': equal, 'token_sha256': row['drafted']['token_sha256'], 'metrics_delta': row['metrics_delta']})
        counts = row['metrics_delta']
        drafted = counts['tensorfold:mtp_drafted_total']
        rounds = counts['tensorfold:decode_rounds_total']
        assert rounds > 0
        if label == 'cap0':
            assert drafted == 0
        if label == 'cap1':
            base_counts = next(x['metrics_delta'] for x in reference['checks'] if x['prompt'] == row['prompt'])
            # The public draft counter includes copied continuations, which do not use the MTP cap.
            assert 0 < drafted < base_counts['tensorfold:mtp_drafted_total']
            assert rounds > base_counts['tensorfold:decode_rounds_total']
    concurrent = json.loads((root / (label + '.bench_concurrent.json')).read_text())
    for cell in concurrent['cells']:
        assert cell['failed'] == 0
        for kind in ('alone', 'serial'):
            assert cell[kind]['unequal'] == cell[kind]['failed'] == 0
            summary['passed'] += cell[kind]['equal']
    speed = json.loads((root / (label + '.bench_openai.json')).read_text())
    run['decode'] = [{'prompt': row['prompt'], 'decode_tps_median': row['decode_tps_median'], 'decode_tps_all': row['decode_tps_all']} for row in speed]
    summary['runs'].append(run)
assert summary['failed'] == 0
(root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
