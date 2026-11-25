# Inkling 2 Mini on Metal as a persistent service (Mac Studio)

The native Metal engine serves `thinkingmachines/Inkling-2-Mini` (`inkling2_loop`, ModelOpt NVFP4 experts,
BF16 trunk) on one Apple Silicon Mac as a persistent, bearer-key-gated OpenAI-compatible endpoint for
authorized OMP agents. It runs beside `drama-speech` (Talia's live voice) on the same GPU: the voice server
stays responsive because the service holds one inference slot and idles between requests.

Verified 2026-11-25 on Mac17,15 (Apple M5 Ultra, 256 GiB, macOS 27.0), `tensorfold-native` 1.3.4 built with
`zig build native -Dcpu=apple_m1`.

## Checkpoint

| Repository | On disk | Notes |
| --- | --- | --- |
| `thinkingmachines/Inkling-2-Mini` | `~/Models/Inkling-2-Mini` (18.8 GiB `model-00001-of-00001.safetensors` + 0.6 GiB `mtp.safetensors`) | The verified checkpoint from card 188/210's Mac work; `info` reports `family: inkling2_loop`, `memory floor: 20.80 GiB` |

## Service

- Binary: this checkout's `zig-out/native/bin/tensorfold-native` (copy it before changing worktrees; the
  recipe uses a copy outside the worktree so later `zig build` runs cannot swap the running binary).
- launchd LaunchAgent `com.infra.tensorfold.inkling` in the `agents` account (this Mac's bot account):
  `~/Library/LaunchAgents/com.infra.tensorfold.inkling.plist`, `KeepAlive`, `ThrottleInterval` 30 s,
  logs to `~/Library/Logs/infra/tensorfold-inkling.log`.
- Endpoint `http://127.0.0.1:8082/v1`, loopback only. Port 8081 stays reserved for `mac-admin model-use`
  (the qwen/glm swap door); do not reuse it.
- Serve flags: `--host 127.0.0.1 --port 8082 --name inkling-2-mini --parallel 1 --context 32768
  --api-key-file /Users/agents/.config/tensorfold/inkling-keys`, with `TF_INKLING2_TRUNK=mxfp8` in the
  environment (card 251's MXFP8 trunk: same tokens as bf16 at measured speed; drop the variable to serve
  the bf16 trunk exactly as stored).
- `--parallel 1` is the concurrency bound: one stream at a time, extras wait. `--context 32768` keeps one
  stream's KV plus prompt state inside the admission budget (startup prints
  `171.6 GiB for stream and prompt state` against 256 GiB).
- Memory bound is admission refusal: the server reports the 20.8 GiB floor and refuses to open the model
  when the host cannot supply it, rather than displacing `drama-speech`.

### API keys

`~/.config/tensorfold/inkling-keys`, `chmod 600`, one `label[scope]: key` per line
([key scopes](../../api.md#key-scopes)):

- `omp-agents[inference,read]`: the key OMP agents send; it can POST chat/completions and GET models.
- `monitor[read]`: health/metrics only; POSTs answer 403 `insufficient_scope`.

Unauthenticated requests answer 401 on every route except `GET /health` (returns only `{"status":"ok"}`).
Replace the file atomically or `kill -HUP <pid>` to rotate keys.

### OMP provider routing

`~/.omp/agent/models.yml` gains an `inkling-local` provider — the same convention as `mac-local`/`spark-local`
but with auth:

```yaml
  inkling-local:
    baseUrl: http://127.0.0.1:8082/v1
    api: openai-completions
    apiKey: "!/usr/bin/grep omp-agents /Users/agents/.config/tensorfold/inkling-keys | /usr/bin/awk '{print $NF}'"
    authHeader: true
    models:
      - { id: inkling-2-mini, name: "Inkling 2 Mini (local)", reasoning: true, input: [text],
          contextWindow: 32768, maxTokens: 8192, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }
```

Agents then address the model as `inkling-local/inkling-2-mini`. Running OMP processes pick up `models.yml`
on their next provider refresh; a brand-new agent sees it immediately.

### Start / restart / rollback

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.infra.tensorfold.inkling.plist   # start
launchctl kickstart -k gui/$(id -u)/com.infra.tensorfold.inkling                            # controlled restart
launchctl bootout gui/$(id -u)/com.infra.tensorfold.inkling && rm ~/Library/LaunchAgents/com.infra.tensorfold.inkling.plist
                                                                                          # rollback: gone, GPU freed
tail -f ~/Library/Logs/infra/tensorfold-inkling.log                                         # startup + request log
curl -s http://127.0.0.1:8082/health                                                        # {"status": "ok"}
```

A crash restarts under `KeepAlive` after the 30 s throttle. Model load is ~7 s (weights mmap'd); the first
request is slower while Metal compiles its pipelines.

## Verification (2026-11-25)

- 401 without a key, 403 for a read-scoped key on POST, 200 for `omp-agents` on `/v1/models` and
  `/v1/chat/completions`; `/v1/models` reports `id: inkling-2-mini`, `owned_by: tensorfold`,
  `context_length: 32768`.
- A real chat request through the bearer key returns `model: "inkling-2-mini"`, `engine: "lanes"`,
  drafts on; the service is still `state = running` afterwards and after `launchctl kickstart -k`.
- `drama-speech` (`127.0.0.1:18794`, `--parallel 1`) answered a voice take in 1.9 s while the Inkling
  service was loaded and idle (18.0 s for the same take earlier that hour with no Inkling; voice takes
  vary with GPU warm state, the service does not hold the GPU between requests).
- The model thinks by default (`reasoning.default_enabled`): a client that wants a short direct answer
  should send `chat_template_kwargs {"enable_thinking": false}` or accept `reasoning_content`.

## Coexistence rules

- `drama-speech` owns voice latency: never raise `--parallel` above 1 and never run GPU benchmarks against
  this service while a voice take may be in flight.
- If the Desk stops `drama-speech` for a GPU window (`/Users/Shared/agent-work/mac-gpu-window`), the
  service keeps running; only benchmark loops wait for the window.
- Port 8082 is this service's; 8081 is `mac-admin`'s swap door, 18794 is the voice server.
