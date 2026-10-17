import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / 'tools'))
from bench_concurrent import stream, PROMPTS

label, binary, commit, *flags = sys.argv[1:]
out = here / 'receipts'
out.mkdir(exist_ok=True)
loopback = socket.gethostbyname('localhost')
base = f'http://{loopback}:8093'
env = dict(os.environ, TF_HEAT_HIGH='85', TF_HEAT_LOW='75')
model = str(Path.home() / 'models/nemotron-3.5-super-vl-nvfp4')
args = [binary, 'serve', model, '--name', 'super', '--no-update-check', '--host', loopback, '--port', '8093', '--prompt-cache-gib', '0', *flags]
stop = threading.Event()
violations = []

def gpu(query):
    return subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader,nounits'], text=True).strip()

def memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    values = {k: int(v.split()[0]) for k, v in fields.items()}
    return values['MemAvailable'], values['SwapTotal'] - values['SwapFree']

assert not gpu('--query-compute-apps=pid'), 'GPU already occupied'
swap0 = memory()[1]
metadata = {'label': label, 'commit': commit, 'gpu': gpu('--query-gpu=name,driver_version'), 'cuda': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True), 'serve_flags': args[3:], 'checkpoint': 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3'}
metadata['serve_flags'] = ['<loopback>' if v == loopback else v for v in metadata['serve_flags']]
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
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('qualification deadline')))

def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.read().decode()

def metrics():
    text = get('/metrics')
    return {line.split()[0]: float(line.split()[1]) for line in text.splitlines() if line.startswith(('tensorfold:mtp_drafted_total ', 'tensorfold:mtp_accepted_total ', 'tensorfold:decode_rounds_total '))}

try:
    for _ in range(240):
        if p.poll() is not None:
            raise RuntimeError('server exited during load')
        try:
            get('/v1/models')
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError('load deadline')
    checks = []
    for item in PROMPTS:
        before = metrics()
        drafted = stream(base, 'super', item, 256, 0, 42)
        after = metrics()
        plain = stream(base, 'super', item, 256, 0, 42, draft=False)
        assert drafted.get('token_sha256') and drafted['token_sha256'] == plain.get('token_sha256'), 'draft/plain mismatch'
        checks.append({'prompt': item['name'], 'drafted': drafted, 'plain': plain, 'metrics_delta': {k: after[k] - before[k] for k in before}})
    metadata['checks'] = checks
    for tool, extra in [('bench_concurrent.py', ['--levels', '1,4', '--alone', '--serial']), ('bench_openai.py', [])]:
        command = [sys.executable, str(here / 'tools' / tool), base, 'super', '--tokens', '256', '--reps', '3', '--temperatures', '0', '--label', label, '--output', str(out / (label + '.' + tool[:-3] + '.json')), *extra]
        with (out / (label + '.' + tool[:-3] + '.log')).open('w') as f:
            subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=600)
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
    text = logfile.read_text()
    text = text.replace(str(Path.home()), '<home>').replace(loopback, '<loopback>')
    logfile.write_text(text)
    metadata['violations'] = violations
    metadata['gpu_idle_after'] = not gpu('--query-compute-apps=pid')
    (out / (label + '.json')).write_text(json.dumps(metadata, indent=2))
    assert metadata['gpu_idle_after'] and not violations
