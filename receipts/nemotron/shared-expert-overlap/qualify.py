"""One GPU hold: serve BINARY, check drafted == draft:false per prompt, then bench_concurrent --alone --serial,
bench_openai and a cold bench_cache prefill; writes receipts/LABEL.*. Usage: qualify.py LABEL BINARY COMMIT TOOLS_DIR"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

label, binary, commit, tools = sys.argv[1:5]
here = Path(__file__).resolve().parent
sys.path.insert(0, tools)
from bench_concurrent import stream, PROMPTS  # noqa: E402

out = here / 'receipts'
out.mkdir(exist_ok=True)
base = 'http://127.0.0.1:8094'
env = dict(os.environ, TF_HEAT_HIGH='85', TF_HEAT_LOW='75')
model = str(Path.home() / 'models/nemotron-3.5-super-vl-nvfp4')
args = [binary, 'serve', model, '--name', 'super', '--no-update-check', '--host', '127.0.0.1', '--port', '8094', '--prompt-cache-gib', '0']
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
binary_sha256 = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
source = {'source_commit': commit, 'binary_sha256': binary_sha256}
metadata = {'label': label, 'commit': commit, 'binary_sha256': binary_sha256, 'gpu': gpu('--query-gpu=name,driver_version'),
            'cuda': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True).strip().splitlines()[-1],
            'serve_flags': ['<loopback>' if v == '127.0.0.1' else v for v in args[3:]],
            'checkpoint': 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3',
            'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
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


def tool(name, extra):
    command = [sys.executable, str(Path(tools) / name), base, 'super', '--output', str(out / f'{label}.{name[:-3]}.json'), *extra]
    with (out / f'{label}.{name[:-3]}.log').open('w') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=900)
    path = out / f'{label}.{name[:-3]}.json'
    data = json.loads(path.read_text())
    for row in data if isinstance(data, list) else [data]:
        row.update(source)
    path.write_text(json.dumps(data, indent=2))


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
        drafted = stream(base, 'super', item, 256, 0, 42)
        plain = stream(base, 'super', item, 256, 0, 42, draft=False)
        assert drafted.get('token_sha256') and drafted['token_sha256'] == plain.get('token_sha256'), 'draft/plain mismatch'
        checks.append({'prompt': item['name'], 'token_sha256': drafted['token_sha256'], 'drafted': drafted, 'plain': plain})
    metadata['checks'] = checks
    run = ['--tokens', '256', '--reps', '3', '--temperatures', '0', '--label', label]
    tool('bench_concurrent.py', [*run, '--levels', '1,4', '--alone', '--serial'])
    tool('bench_openai.py', run)
    prompts = json.loads((Path('<prompts.json>')).read_text())
    doc = next(x for x in prompts['prompts'] if x['name'] == 'long_document')
    req = urllib.request.Request(base + '/tokenize', data=json.dumps({'messages': doc['messages'], 'chat_template_kwargs': {'enable_thinking': False}}).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        ids = json.load(r)['tokens']
    ids_file = out / f'{label}.long_document.ids.json'
    ids_file.write_text(json.dumps(ids))
    tool('bench_cache.py', ['--tokens', '16', '--repeats', '2', '--streams', '1', '--prompt-file', str(ids_file), '--expected-prompt-tokens', str(len(ids)), '--require-cold'])
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
