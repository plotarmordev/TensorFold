"""One GPU hold for the PR 4 follow-up: serve BINARY with its own HOME (the window-cost cache under it), run MODE.
Usage: hold.py LABEL BINARY COMMIT TOOLS_DIR CACHE_HOME MODE; MODE: start | concurrent | turns | fresh"""
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

label, binary, commit, tools, cache_home, mode = sys.argv[1:7]
here = Path(__file__).resolve().parent
sys.path.insert(0, tools)
from bench_evidence import Evidence  # noqa: E402

out = here / 'receipts'
out.mkdir(exist_ok=True)
base = 'http://127.0.0.1:8094'
model = str(Path.home() / 'models/nemotron-3.5-super-vl-nvfp4')
document = json.loads((here / 'long_document.messages.json').read_text())  # the 8,186-token chat prompt's messages
question = 'List the three most important facts above, one sentence each.'
cache_flags = [] if mode in ('turns', 'fresh') else ['--prompt-cache-gib', '0']
args = [binary, 'serve', model, '--name', 'super', '--no-update-check', '--host', '127.0.0.1', '--port', '8094', *cache_flags]
client_env = dict(os.environ, TF_HEAT_HIGH='85', TF_HEAT_LOW='75')
env = dict(client_env, HOME=cache_home)
costs = Path(cache_home) / '.cache/tensorfold/window-costs'
stop = threading.Event()
violations = []


def gpu(query):
    return subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader,nounits'], text=True).strip()


def memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    values = {k: int(v.split()[0]) for k, v in fields.items()}
    return values['MemAvailable'], values['SwapTotal'] - values['SwapFree']


def tables():
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(costs.glob('*.bin'))} if costs.is_dir() else {}


assert not gpu('--query-compute-apps=pid'), 'GPU already occupied'
swap0 = memory()[1]
binary_sha256 = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
source = {'source_commit': commit, 'binary_sha256': binary_sha256}
metadata = {'label': label, 'mode': mode, **source, 'gpu': gpu('--query-gpu=name,driver_version'),
            'serve_flags': ['<loopback>' if v == '127.0.0.1' else v for v in args[3:]],
            'checkpoint': 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3',
            'cost_tables_before': tables(), 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
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


def tool(name, extra):
    path = out / f'{label}.{name[:-3]}.json'
    command = [sys.executable, str(Path(tools) / name), base, 'super', '--output', str(path), *extra]
    with (out / f'{label}.{name[:-3]}.log').open('w') as f:
        code = subprocess.run(command, env=client_env, stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
    data = json.loads(path.read_text())
    for row in data if isinstance(data, list) else [data]:
        row.update(source)
    path.write_text(json.dumps(data, indent=2))
    return code


def chat(messages, draft=True):
    """One greedy 256-token chat reply: its text (kept out of the receipt) and the counters and hash the server reports."""
    body = {'model': 'super', 'messages': messages, 'max_tokens': 256, 'temperature': 0, 'seed': 1234, 'stream': True,
            'stream_options': {'include_usage': True}, 'ignore_eos': True, 'chat_template_kwargs': {'enable_thinking': False}}
    if not draft:
        body['draft'] = False
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    evidence, usage, runtime, text, complete = Evidence(), {}, {}, [], False
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if line == 'data: [DONE]':
                complete = True
                break
            if line.startswith('data:'):
                chunk = json.loads(line[5:])
                usage = chunk.get('usage') or usage
                runtime = chunk.get('tensorfold') or runtime
                for choice in chunk.get('choices', []):
                    evidence.add(choice)
                    text.append((choice.get('delta') or {}).get('content') or '')
    seen = evidence.finish(usage, runtime, complete)
    return ''.join(text), {k: seen[k] for k in ('prompt_tokens', 'cached_tokens', 'cache_state', 'tokens', 'complete', 'token_sha256')}


def second_turn():
    reply = (here / 'turn1.reply.txt').read_text()
    return [*document, {'role': 'assistant', 'content': reply}, {'role': 'user', 'content': question}]


try:
    for _ in range(240):
        if p.poll() is not None:
            raise RuntimeError('server exited during load')
        try:
            urllib.request.urlopen(base + '/v1/models', timeout=10).read()
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError('load deadline')
    if mode == 'concurrent':
        assert tool('bench_concurrent.py', ['--tokens', '256', '--reps', '3', '--temperatures', '0', '--levels', '1,4', '--alone', '--serial', '--label', label]) == 0
    elif mode == 'turns':
        reply, first = chat(document)
        (here / 'turn1.reply.txt').write_text(reply)
        rows = [{'request': 'turn 1, fresh server', **first}]
        _, row = chat(second_turn())
        rows.append({'request': 'turn 2, resumed from turn 1', **row})
        _, row = chat(second_turn(), draft=False)
        rows.append({'request': 'turn 2 again, draft false', **row})
        _, row = chat(document)
        rows.append({'request': 'turn 1 again', **row})
        metadata['requests'] = rows
    elif mode == 'fresh':
        _, row = chat(second_turn())
        metadata['requests'] = [{'request': 'turn 2, fresh server', **row}]
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
    metadata['cost_tables_after'] = tables()
    metadata['violations'] = violations
    metadata['gpu_idle_after'] = not gpu('--query-compute-apps=pid')
    (out / (label + '.json')).write_text(json.dumps(metadata, indent=2))
    assert metadata['gpu_idle_after'] and not violations
