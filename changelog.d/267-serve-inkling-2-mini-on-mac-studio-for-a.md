# Serve Inkling 2 Mini on the Mac Studio for authorized OMP agents

- `docs/recipes/inkling-2-mini-metal/README.md` records the resident Inkling 2 Mini service running on the
  Mac Studio (M5 Ultra, 256 GiB): `tensorfold-native` 1.3.4 (`zig build native -Dcpu=apple_m1`, installed at
  `~/.local/libexec/tensorfold-inkling/`) serves `~/Models/Inkling-2-Mini` on `127.0.0.1:8082` under launchd
  `com.infra.tensorfold.inkling` in the agents account, with `TF_INKLING2_TRUNK=mxfp8`, `--parallel 1` and
  `--context 32768` as the concurrency and memory bounds, and a scoped bearer-key file
  (`omp-agents[inference,read]`, `monitor[read]`) as the authorization layer.
- `~/.omp/agent/models.yml` gained an `inkling-local` provider (`baseUrl http://127.0.0.1:8082/v1`,
  `apiKey` read from the key file at runtime, `authHeader`), so authorized OMP agents address the model as
  `inkling-local/inkling-2-mini` — a local provider in the shape of `mac-local`/`spark-local`, plus the
  bearer key the new service enforces, without touching the swap door.
- Verified on the host: unauthenticated requests 401, a read-scoped key 403s on POST, an authorized chat
  request returns `model: inkling-2-mini`, and a second authorized OMP process reached the model through
  the `inkling-local` provider (omp `-p` requests logged `key=omp-agents`, a reviewer bot's provider call
  returned `OK`). The service stays `state = running` after requests and after `launchctl kickstart -k`.
  `drama-speech` (Talia's voice, `:18794`) answered a take in 1.9 s with the service loaded and idle; the
  arrangement was approved by the Desk before install, leaving taku's services and the `:8081` model door
  untouched.
