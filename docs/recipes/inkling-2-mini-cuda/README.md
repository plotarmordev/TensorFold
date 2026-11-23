# Inkling 2 Mini on TOP's GB10

`run-script.sh` runs thinkingmachines/Inkling-2-Mini (revision `ed40e7da62`) whole on TOP's Spark through the Desk's
GPU door. The job has no network, so the checkpoint must already sit in the pilot folder as `Inkling-2-Mini` (or
`models/Inkling-2-Mini`); the script refuses and lists the Inkling configs under `/assets` when it does not.

The server is jschmied's #4441 engine (`inkling2.native`, model type `inkling2_loop`, on his #4364 and #4365): graphs,
fused input projections and one batched forward for every stream's rows in a round. #4387's checks run against it.
The script fails on any of: `tf-cuda-test kernel-contract "inkling2 loop"` (the served message and gate kernels
against FP64 references, with their planted controls), `tf-cuda-test inkling2-serving` (#4387's checks), `tf-cuda-test inkling2-mini-ref` (32
greedy tokens equal to the float64 host reference), that reference's planted control staying green, a failed stream
in `tools/bench_concurrent.py` at 1 and 4 streams, or 4 streams giving under 1.5 times one stream's total tok/s. The
receipt also sets one stream's decode against GB10's bandwidth limit: the weight bytes a token reads (from the
checkpoint's headers) over 273 GB/s.

## Receipt: TOP's GB10, one capped run

Job `top-16bffb6ca094` at `9f542e6fa`, `--context 8192`, greedy, 256 tokens, 3 reps, medians. This is one run with
these settings, not a benchmark.

| Streams | First token (code / chat) | Prompt (code) | Decode a stream | Total decode | Failed |
|---|---|---|---|---|---|
| 1 | 0.077 s / 0.038 s | 195.9 tok/s | 32.5 tok/s | 32.5 tok/s | 0 |
| 4 | 0.189 s / 0.092 s | 82.6 tok/s | 29.6 tok/s | 117.8 tok/s (3.62x) | 0 |

Memory, sampled after each step (not process peaks): host used (`/proc/meminfo`) was 10.9 GiB before the server,
31.6 GiB with Mini loaded for 4 streams, and the same after both levels; the server held 20.0 GiB of GPU memory
(`nvidia-smi`).

Bandwidth limit: a decoded token reads about 5.55 GB of weights (core layers 3 passes, 8 of 128 experts a visit, the
whole unembedding), so 273 GB/s caps one stream near 49 tok/s; 32.5 tok/s is 66% of it. #4387's own engine, removed
here, gave 5.7 tok/s at one stream and 9.4 at four on the same GB10 (job `top-0759755a0515`).

Job `top-9defa53c0bfc` at `2a37918af` reran it with the planted control's new exit: the control's 6 of 32 differing
tokens exit non-zero and the script goes on, ending 0 with 33.8 tok/s at one stream and 121.4 at four (3.59x).

## RTX Spark (N1X): native Windows ARM64 on Net's laptop

`n1x-run.ps1` runs the same checkpoint (`C:\models\Inkling-2-Mini-ed40e7da62`) on the laptop through its Codync bot
`codync@rtx-spark`, relayed by the Desk, one step a run. It installs and downloads nothing; the laptop has PowerShell
but no Python or WSL, so the 1/4-stream client is the script's own, not `bench_concurrent.py`.

1. `-Step fatbins`, then `-Step build`: `zig build fatbins`, then `zig build install native`, both `-Dsm=121
   -Doptimize=ReleaseFast` with the CUDA 13.5 `windows-arm64` nvcc and `cl.exe` arm64 (`vcvarsall arm64`). 189's
   Windows fixes are on this branch (`__noinline__` in `glm_index_rope.cuh`, `/wd4819`, POSIX file modes and COM
   `@alignCast` in the server, no `tf-muse`). Each stops at 9 min; a cut-off step resumes from the zig cache.
2. `-Step check`: 188's GB10 checks unchanged, so the laptop's output is held to the host reference the same way:
   `kernel-contract "inkling2 loop"`, `inkling2-serving` and `inkling2-mini-ref` (32 greedy tokens equal to the float64
   reference).
3. `-Step control`: the reference's planted control, which must exit non-zero with its tokens differing.
4. `-Step serve`: one server at `--parallel 4 --context 8192`, greedy 256-token replies at 1 and 4 streams; first token
   and decode per stream come from the server's own `done` lines, with the ready time, the WDDM pool's peak
   (`nvidia-smi -lms 500` in its own process) and the server's peak working set.

What differs from the GB10 and is not copied from it:

- `nvidia-smi` there also lists an NVIDIA NPU, so the script finds the N1X by name and queries only its index;
  `memory.used` over every device gave an array and stopped run 1's serve step.
- A cold build took 647 s (run 1) and 901 s (run 2) in one piece, over a run's 10 minutes, so nvcc's kernel images are
  their own step. The host float64 reference takes 226 s on the laptop, so the planted control is its own step: run 2
  ran both in one step for 11 min 10 s.
- The serve step times replies from the server's log, not the client: run 2's client read the stream line by line and
  spawned `nvidia-smi` per line, which put it at 6.7 tok/s at one stream while the server decoded at 32.2.
- Memory is the WDDM pool the firmware gives CUDA (51,816 MiB on this laptop), not `/proc/meminfo`; admission reads it
  through `cuMemGetInfo_v2`. The 20 GB checkpoint leaves about 20 GiB of it for the cache and four streams.
- The power cap (42-49 W at about 2.1 GHz in HP Performance mode, near 1.4 GHz in power saving, from the 2026-11-04
  receipt in [WINDOWS.md](../../WINDOWS.md)) bounds decode, so each run prints the power plan, limit and SM clock.

Run 1 at `b2ddc3de` (clean tree): build exit 0; check passed as in run 2; serve was ready in 55.3 s, then the NPU row
above stopped the script before any tok/s.

### Receipt: run 2 at `2a37918af`, one capped run

Desk request `bb869e9b` on `codync@rtx-spark`, clean clone (`git status --porcelain` empty before and after the build),
`powershell -ExecutionPolicy Bypass -File docs\recipes\inkling-2-mini-cuda\n1x-run.ps1 -Step <step>`. This is one run
with these settings, not a benchmark.

| Step | Exit | Time | Result |
|---|---|---|---|
| `build` | 0 | 15 min 1 s (cold cache) | no error lines |
| `check` (with the control, as it then was) | 0 | 11 min 10 s | `kernel-contract "inkling2 loop"` 5/5 pass gates 1-4, 2/2 controls red; `inkling2-serving` PASS (CUDA equals the host reference token for token, fresh equals resumed, admission); `inkling2-mini-ref` 0 of 32 tokens differ (smallest margin 0.110, 226 s); planted control 6 of 32 differ, exit 1 |
| `serve` | 0 | | ready in 58.4 s: 19.82 GiB model, 4 streams x 0.49 GiB at 8192, 16 GiB prompt cache |

| Streams | First token | Decode a stream | Total decode | Failed |
|---|---|---|---|---|
| 1 | 0.104 s | 32.2 tok/s | 32.2 tok/s | 0 |
| 4 | 0.096-0.357 s | 28.8 / 29.1 / 29.4 / 29.7 tok/s | 117.0 tok/s (3.63x) | 0 |

Times are the server's `done` lines (256 tokens each, `finish=length`). Peak GPU memory: the WDDM pool's largest
sample was 23,335 MiB (run 1 sampled 2.7 GiB before the server). The N1X decoded one stream as fast as GB10 (32.2
against 32.5 tok/s) and four streams at 117.0 against 117.8.
