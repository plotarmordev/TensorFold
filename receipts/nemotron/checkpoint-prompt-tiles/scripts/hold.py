"""One GPU hold on a fresh server; writes receipts/LABEL.*. Usage: hold.py MODE LABEL BINARY COMMIT TOOLS_DIR [FRESH]

MODE decode: the default window; drafted == draft:false per bench_concurrent prompt, bench_concurrent --alone --serial,
bench_openai. MODE prefill (2k, 8k, 32k, 64k) and long (128k): a 131,584-token window, each token array cold through
bench_cache --require-cold, then the same array with "draft": false against the drafted reply. FRESH=1 deletes this
binary's window-cost cache entries first (its first boot), so it measures its own table.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request

mode, label, binary, commit, tools = sys.argv[1:6]
fresh = len(sys.argv) > 6 and sys.argv[6] == '1'
here = Path(__file__).resolve().parent
sys.path.insert(0, tools)
from bench_concurrent import stream, PROMPTS  # noqa: E402

SIZES = {'decode': [], 'prefill': [2048, 8192, 32768, 65536], 'long': [131072]}[mode]
CONTEXT = 131584
out = here / 'receipts'
out.mkdir(exist_ok=True)
base = 'http://127.0.0.1:8097'
env = dict(os.environ, TF_HEAT_HIGH='85', TF_HEAT_LOW='75')
model = str(Path.home() / 'models/nemotron-3.5-super-vl-nvfp4')
args = [binary, 'serve', model, '--name', 'super', '--no-update-check', '--host', '127.0.0.1', '--port', '8097', '--prompt-cache-gib', '0']
if SIZES:
    args += ['--context', str(CONTEXT)]
stop = threading.Event()
violations = []


def gpu(query):
    return subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader,nounits'], text=True).strip()


def memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    values = {k: int(v.split()[0]) for k, v in fields.items()}
    return values['MemAvailable'], values['SwapTotal'] - values['SwapFree']


def cost_cache(context):
    """The engine's window cost cache file for this binary at a window (cuda_costs.zig's key, cuda_engine's max_len)."""
    version = ctypes.c_int()
    ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(version))
    max_len = (context + 16 + 511) // 512 * 512
    shape = f'sm121 cuda{version.value} 88/4096/131072/512/{max_len} draft131072 own modelopt'.encode()
    h = hashlib.sha256()
    for part in (Path(binary).read_bytes(), b'nemotron-cuda', b'NVIDIA GB10', shape):
        h.update(struct.pack('<Q', len(part)))
        h.update(part)
    return Path.home() / '.cache/tensorfold/window-costs' / (h.hexdigest() + '.bin')


def corpus(sections):
    """make_prompts.py's field log (fixed seed): varied, numeric text, about 136 tokens a section."""
    stations = ["Alder Ridge", "Brennan Weir", "Cold Spring", "Dunmore Lift", "East Fen", "Fallow Creek",
                "Garrow Basin", "Hollin Cut", "Ivel Reach", "Juniper Hill"]
    crew = ["Okafor", "Lindqvist", "Marchetti", "Tanaka", "Dubois", "Haddad", "Novak", "Reyes", "Sørensen", "Chen"]
    events = [
        "the intake screen was found partly blocked by leaf litter and cleared by hand",
        "pump {p} tripped on overcurrent and was restarted after the bearing temperature fell below {t} degrees",
        "turbidity rose to {n} NTU after overnight rain and the coagulant dose was raised by {d} percent",
        "a contractor reported a weeping joint on the {m} mm main, which was clamped the same afternoon",
        "the chlorine residual at the outlet held at {c} mg/L throughout the shift",
        "telemetry from the reservoir float dropped out for {k} minutes; the radio link was reseated",
        "a resident called about low pressure on the upper streets; the valve at the hill junction was found {o} percent open",
        "the standby generator ran its weekly test for {k} minutes with no faults",
        "sludge was drawn from settling tank {p} and the blanket depth measured {r} metres afterwards",
        "the flow meter on the transfer main was compared against the portable unit and read {e} percent high",
    ]
    notes = ["Readings were logged at the usual intervals.", "No complaints were received.",
             "The crew noted that the access track is rutted and will need grading before winter.",
             "Spare filters were counted and two boxes were reordered.",
             "The night operator asked that the alarm thresholds be reviewed at the next meeting.",
             "Algae were visible along the north margin of the reservoir but samples were clear.",
             "The site gate lock was replaced after it seized in the cold."]
    rng = random.Random(20261016)
    text = ["HARROW VALLEY WATER AUTHORITY — CONSOLIDATED FIELD LOG\n"]
    for i in range(sections):
        day, month = 1 + i % 28, 1 + (i // 28) % 12
        station = rng.choice(stations)
        lead, second = rng.sample(crew, 2)
        text.append(f"\nEntry {i + 1}: {station}, 2025-{month:02d}-{day:02d}, shift lead {lead}, with {second}.\n")
        text.append(f"Flow averaged {rng.randint(180, 950)} m3/h and the wet well level was {rng.uniform(2.1, 9.8):.2f} m at handover. ")
        for _ in range(rng.randint(2, 4)):
            event = rng.choice(events).format(
                p=rng.randint(1, 4), t=rng.randint(55, 80), n=rng.randint(3, 40), d=rng.randint(5, 30),
                m=rng.choice([150, 225, 300, 450]), c=f"{rng.uniform(0.4, 1.6):.2f}", k=rng.randint(3, 45),
                o=rng.randint(10, 90), r=f"{rng.uniform(0.3, 1.9):.1f}", e=f"{rng.uniform(0.5, 6.0):.1f}")
            text.append(f"During the shift {event}. ")
        text.append(rng.choice(notes) + "\n")
    return "".join(text)


def post(path, body, timeout=60):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ids_file(size):
    """The token array of `size` ids: a head naming the size, then the field log (written once, then reused)."""
    path = here / f'ids-{size}.json'
    if not path.exists():
        head = post('/tokenize', {'prompt': f'Cold prefill of {size} tokens.\n', 'add_special_tokens': True})['tokens']
        body = post('/tokenize', {'prompt': corpus(1100), 'add_special_tokens': False}, timeout=600)['tokens']
        assert len(head) + len(body) >= size, 'corpus too short'
        path.write_text(json.dumps((head + body)[:size]))
    return path


def heat_waits(start=0):
    """Seconds of the server's logged heat waits (`heat_wait_s`) past byte `start` of its log."""
    with logfile.open('rb') as f:
        f.seek(start)
        return round(sum(float(x) for x in re.findall(rb'heat_wait_s ([0-9.]+)', f.read())), 1)


assert not gpu('--query-compute-apps=pid'), 'GPU already occupied'
if fresh:
    for context in (16384, CONTEXT):
        cost_cache(context).unlink(missing_ok=True)
swap0 = memory()[1]
metadata = {'mode': mode, 'label': label, 'commit': commit, 'binary_sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
            'build': (Path(binary).parent / 'build').read_text().splitlines(),
            'gpu': gpu('--query-gpu=name,driver_version'),
            'cuda': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True).strip().splitlines()[-1],
            'serve_flags': ['<loopback>' if v == '127.0.0.1' else v for v in args[3:]],
            'checkpoint': 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3',
            'costs_cache_deleted_before_boot': fresh, 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
source = {'source_commit': commit, 'binary_sha256': metadata['binary_sha256']}
logfile = out / (label + '.server.log')
log = logfile.open('w')
p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)


def guard():
    with (out / (label + '.guard.jsonl')).open('w') as f:
        while not stop.wait(5):
            temp = int(gpu('--query-gpu=temperature.gpu'))
            avail, swap = memory()
            row = {'temperature_c': temp, 'available_kib': avail, 'swap_growth_kib': swap - swap0}
            f.write(json.dumps(row) + '\n')
            f.flush()
            if temp > 90 or avail < 6291456 or swap - swap0 > 4194304:
                violations.append(row)
                p.terminate()
                return


thread = threading.Thread(target=guard)
thread.start()
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('hold deadline')))


def tool(name, extra, tag=''):
    stem = f'{label}.{name[:-3]}{tag}'
    command = [sys.executable, str(Path(tools) / name), base, 'super', '--output', str(out / f'{stem}.json'), *extra]
    with (out / f'{stem}.log').open('w') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=1800)
    path = out / f'{stem}.json'
    data = json.loads(path.read_text())
    for row in data if isinstance(data, list) else [data]:
        row.update(source)
    path.write_text(json.dumps(data, indent=2))
    return data


try:
    started = time.perf_counter()
    for _ in range(300):
        if p.poll() is not None:
            raise RuntimeError('server exited during load')
        try:
            urllib.request.urlopen(base + '/v1/models', timeout=10).read()
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError('load deadline')
    metadata['ready_s'] = round(time.perf_counter() - started, 1)
    metadata['admission'] = re.sub(r'^\[tensorfold\] ', '', next(x for x in logfile.read_text().splitlines() if 'GiB each' in x))
    if mode == 'decode':
        checks = []
        for item in PROMPTS:
            drafted = stream(base, 'super', item, 256, 0, 42)
            plain = stream(base, 'super', item, 256, 0, 42, draft=False)
            assert drafted.get('token_sha256') and drafted['token_sha256'] == plain.get('token_sha256'), 'draft/plain mismatch'
            checks.append({'prompt': item['name'], 'token_sha256': drafted['token_sha256'], 'drafted': drafted, 'plain': plain})
        metadata['checks'] = checks
        run = ['--tokens', '256', '--reps', '3', '--temperatures', '0', '--label', label]
        tool('bench_concurrent.py', [*run, '--levels', '1,4', '--alone', '--serial'])
        tool('bench_openai.py', run)
    prompts = []
    for size in SIZES:
        path = ids_file(size)
        mark = logfile.stat().st_size
        cache = tool('bench_cache.py', ['--tokens', '16', '--repeats', '1', '--streams', '1', '--prompt-file', str(path), '--expected-prompt-tokens', str(size), '--require-cold'], f'-{size}')
        waited = heat_waits(mark)
        plain = stream(base, 'super', {'name': f'ids-{size}', 'kind': 'completion', 'prompt': json.loads(path.read_text())}, 16, 0, 1234, draft=False)
        first = cache['rows'][0]
        prompts.append({'prompt_tokens': size, 'ids_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'qualified': cache['qualified'],
                        'ttft_s': [row.get('ttft_s') for row in cache['rows']], 'cached_tokens': [row.get('cached_tokens') for row in cache['rows']],
                        'heat_wait_s': waited, 'drafted_token_sha256': first.get('token_sha256'), 'plain_token_sha256': plain.get('token_sha256'),
                        'plain_cached_tokens': plain.get('cached_tokens'), 'drafted_equals_plain': first.get('token_sha256') == plain.get('token_sha256')})
    metadata['prompts'] = prompts
    metadata['costs_file_present'] = cost_cache(CONTEXT if SIZES else 16384).exists()
    metadata['heat_wait_s'] = heat_waits()
    metadata['result'] = 'completed'
finally:
    p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    stop.set()
    thread.join()
    log.close()
    logfile.write_text(logfile.read_text().replace(str(Path.home()), '<home>').replace('127.0.0.1', '<loopback>'))
    metadata['violations'] = violations
    metadata['gpu_idle_after'] = not gpu('--query-compute-apps=pid')
    (out / (label + '.json')).write_text(json.dumps(metadata, indent=2))
    assert metadata['gpu_idle_after'] and not violations
    assert all(x['qualified'] and x['drafted_equals_plain'] for x in metadata.get('prompts', [])), 'prefill check failed'
