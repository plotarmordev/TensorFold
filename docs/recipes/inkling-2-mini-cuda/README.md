# Inkling 2 Mini on TOP's GB10

`run-script.sh` runs thinkingmachines/Inkling-2-Mini (revision `ed40e7da62`) whole on TOP's Spark through the Desk's
GPU door. The job has no network, so the checkpoint must already sit in the pilot folder as `Inkling-2-Mini` (or
`models/Inkling-2-Mini`); the script refuses and lists the Inkling configs under `/assets` when it does not.

The server is jschmied's #4441 engine (`inkling2.native`, model type `inkling2_loop`, on his #4364 and #4365): graphs,
fused input projections and one batched forward for every stream's rows in a round. #4387's checks run against it.
Upstream, jschmied closed #4441 on 23 Nov as superseded: Mini on CUDA goes through #4387's engine there (#4640, then
#4697, #4718, #4728 and #4729 for speed). This branch is local; an upstream claim must move to that stack first.
The script fails on any of: `tf-cuda-test kernel-contract "inkling2 loop"` (the served message and gate kernels
against FP64 references, with their planted controls), `tf-cuda-test inkling2-serving` (#4387's checks), `tf-cuda-test inkling2-mini-ref` (32
greedy tokens equal to the float64 host reference), that reference's planted control staying green, a failed stream
in `tools/bench_concurrent.py` at 1 and 4 streams, or 4 streams giving under 1.5 times one stream's total tok/s. The
receipt also sets one stream's decode against GB10's bandwidth limit: the weight bytes a token reads (from the
checkpoint's headers) over 273 GB/s.

## Receipt: TOP's GB10, one capped run

Job `top-0b05213c9f06` at `4a5089e70`, `--context 8192`, greedy, 256 tokens, 3 reps, medians; the build ran first, then
every check and both levels inside `timeout 600`. This is one run with these settings, not a benchmark.

| Streams | First token (code / chat) | Prompt (code) | Decode a stream (code / chat) | Total decode (code / chat) | Failed |
|---|---|---|---|---|---|
| 1 | 0.075 s / 0.037 s | 199.4 tok/s | 32.9 / 32.8 tok/s | 32.9 / 32.8 tok/s | 0 |
| 4 | 0.186 s / 0.091 s | 83.7 tok/s | 29.8 / 29.8 tok/s | 117.9 / 118.8 tok/s | 0 |

The 4-stream gate takes each level's better prompt: 118.8 over 32.9 tok/s, 3.61x. Every stream sends the same prompt,
so the four streams route to the same experts; mixed traffic may give a lower 4-stream rate.

Checks in the same job: the CUDA engine matched the host reference token for token, fresh and resumed prompts gave
identical logits and tokens, and two lanes in shared rounds kept their tokens; 0 of 32 greedy tokens differed from
the float64 reference (smallest reference margin 0.11), and its planted control (no messages) differed at 6. The
kernel contract passed gates 1-4 on all 5 loop kernels with both planted controls red.

Memory, sampled after each step (not process peaks): host used (`/proc/meminfo`) was 11.1 GiB before the server,
31.8 GiB with Mini loaded for 4 streams and 32.2 GiB after both levels; the server held 20.0 GiB of GPU memory
(`nvidia-smi`).

Bandwidth limit: a decoded token reads about 5.55 GB of weights (core layers 3 passes, 8 of 128 experts a visit, the
whole unembedding), so 273 GB/s caps one stream near 49 tok/s; 32.9 tok/s is 67% of it. #4387's own engine, removed
here, gave 5.7 tok/s at one stream and 9.4 at four on the same GB10 (job `top-0759755a0515`).

Job `top-9defa53c0bfc` at `2a37918af` reran it with the planted control's new exit: the control's 6 of 32 differing
tokens exit non-zero and the script goes on, ending 0 with 33.8 tok/s at one stream and 121.4 at four (3.59x).

Job `top-3546b0a66f9b` at `abc67b2d` ran the planted control after the reference instead of beside it: 0 and 6 of 32
differed as before (154.9 s and 150.8 s), 32.9 tok/s at one stream, then the 600 s window closed in the 4-stream
bench. The script runs its one planted control beside the reference.

## RTX Spark (N1X): native Windows ARM64 on Net's laptop

`n1x-run.ps1` runs the same checkpoint (`C:\models\Inkling-2-Mini-ed40e7da62`) on the laptop through its Codync bot
`codync@rtx-spark`, relayed by the Desk, one step a run. It installs and downloads nothing; the laptop has PowerShell
but no Python or WSL, so the 1/4-stream client is the script's own, not `bench_concurrent.py`.

1. `-Step fatbins`, `-Step images`, `-Step runner`, then `-Step server`: `zig build fatbins`, `zig build
   kernel-contract-fatbins`, `zig build tf-cuda-test`, then `zig build native`, each `-Dsm=121 -Doptimize=ReleaseFast`
   with the CUDA 13.5 `windows-arm64` nvcc and `cl.exe` arm64 (`vcvarsall arm64`). 189's Windows fixes are on this
   branch (`__noinline__` in `glm_index_rope.cuh`, `/wd4819`, POSIX file modes and COM `@alignCast` in the server, no
   `tf-muse`). Each stops at 9 min and prints its log's tail and the images installed; a cut-off step resumes from
   the zig cache.
2. `-Step check`: 188's GB10 checks unchanged, so the laptop's output is held to the host reference the same way:
   `kernel-contract "inkling2 loop"`, `inkling2-serving` and `inkling2-mini-ref` (32 greedy tokens equal to the float64
   reference).
3. `-Step control`: the reference's planted control, which must exit non-zero with its tokens differing.
4. `-Step serve`: one server at `--parallel 4 --context 8192`, greedy 256-token replies at 1 and 4 streams; first token
   and decode per stream come from the server's own `done` lines, with the ready time, the WDDM pool's peak
   (`nvidia-smi -lms 500` in its own process) and the server's peak working set.

Steps 2-4 share the builds' 9-minute deadline: each `tf-cuda-test` call, the 240 s server-load wait and each request
is bounded by the step's remaining time, and a killed call prints its log's tail like a build step's.

What differs from the GB10 and is not copied from it:

- `nvidia-smi` there also lists an NVIDIA NPU, so the script finds the N1X by name and queries only its index;
  `memory.used` over every device gave an array and stopped run 1's serve step.
- A cold build took 647 s (run 1) and 901 s (run 2) in one piece, over a run's 10 minutes, so nvcc's kernel images are
  their own step; `install` also builds a dozen products this card never runs (run 4 below), so the runner and the
  server are a step each, and the 37 images only `tf-cuda-test` embeds (DeepSeek's EXL3 units, the contract's
  controls, H3) are a step before the runner (run 5 below). The host float64 reference takes 226 s on the laptop, so
  the planted control is its own step: run 2 ran both in one step for 11 min 10 s.
- The serve step times replies from the server's log, not the client: run 2's client read the stream line by line and
  spawned `nvidia-smi` per line, which put it at 6.7 tok/s at one stream while the server decoded at 32.2.
- Memory is the WDDM pool the firmware gives CUDA (51,816 MiB on this laptop), not `/proc/meminfo`; admission reads it
  through `cuMemGetInfo_v2`. The 20 GB checkpoint leaves about 20 GiB of it for the cache and four streams.
- The power cap (42-49 W at about 2.1 GHz in HP Performance mode, near 1.4 GHz in power saving, from the 2026-11-04
  receipt in [WINDOWS.md](../../WINDOWS.md)) bounds decode, so each run prints the power plan, limit and SM clock.

Run 1 at `b2ddc3de` (clean tree): build exit 0; check passed as in run 2; serve was ready in 55.3 s, then the NPU row
above stopped the script before any tok/s.

Run 2 at `2a37918a` (request `bb869e9b`, clean tree): one 15-minute `build` and one `check` with the control (11 min
10 s), both exit 0 with run 6's results; serve 32.2 tok/s at one stream, 117.0 total at four, pool peak 23,335 MiB.

Run 3 at `a86ccbd1` (request `39062c42`, clean tree): `-Step fatbins` exited 1 in 20 s with `no such step:
fatbins-Dsm=121`, as PowerShell unrolled the one-element target list to a string that `+` then appended to; the
script now keeps it an array.

Run 4 at `26de8309` (request `407e1d7b`, clean tree): `-Step fatbins` exited 0 in 83 s with 143 images; `-Step build`
(`install native`) stopped at 9 min twice without a compile error, building `tf-drama-quant`, `tf-qwen35-learn` and
the rest of `install` beside the two binaries the checks and serving run. Cold aarch64-windows ReleaseFast
cross-compiles on a Mac took 104 s for `tf-cuda-test` and 129 s for `tensorfold-native`, each on one core.

Run 5 at `2ce7e54c` (request `4fbd4d3b`, clean tree): `-Step fatbins` exited 0 in 122 s with nothing changed since
run 4's, and `-Step runner` stopped at 9 min twice with no output, while `-Step server` exited 0 in 449 s. Every
warm build had run nvcc on every image again: each image's cache key held a side-effect `nvcc --version` capture,
rewritten each build, so no step resumed. Images are now keyed by nvcc's own bytes (a second `fatbins
kernel-contract-fatbins` with a stand-in nvcc makes 0 of its 180 calls), the runner's 37 nvcc-only images are their
own step, and each build step logs zig's output. Serve passed again: ready in 55.2 s, 32.2 tok/s at one stream (first
token 0.098 s), 115.4 total at four, pool peak 23,119 MiB; check and control did not run without `tf-cuda-test`.

### Receipt: run 6 at `2c9228a8`, one capped run

Desk run on `codync@rtx-spark`, relayed by Capri, 24 Nov 13:56:50-14:27:27, warm folder `tf243-a86c`, clean tree.
`vcvarsall arm64` 0, cl 14.44.35207 HostARM64, nvcc 13.5 V13.5.37, zig 0.17.0; N1X sm_121, 48 SMs, 51,816 MiB,
driver 621.01, power plan Balanced. Each step is
`powershell -ExecutionPolicy Bypass -File docs\recipes\inkling-2-mini-cuda\n1x-run.ps1 -Step <step>`. This is one run
with these settings, not a benchmark.

| Step | Exit | Time | Result |
|---|---|---|---|
| `fatbins` | 0 | 84 s | 143 images |
| `images` | 0 | 151 s | 180 images |
| `runner` | 0 | 493 s | `tf-cuda-test` |
| `server` | 0 | 347 s | `tensorfold-native` |
| `check` | 0 | | `kernel-contract "inkling2 loop"` 5/5 pass gates 1-4, 2/2 controls red, 0 limit problems (101,376 B shared, 1024 threads; MLP128 sm_120 only); `inkling2-serving` PASS (CUDA equals the host reference token for token, fresh equals resumed, 2 lanes share rounds and a 3rd waits at 17.60 GiB weights); `inkling2-mini-ref` 0 of 32 greedy tokens differ (smallest margin 0.110) |
| `control` | 0 | | planted control 6 of 32 differ (positions 15, 18, 19, 38, 39, 42), its exit non-zero as required |
| `serve` | 0 | | ready in 58.5 s |

| Streams | First token | Decode a stream | Total decode | Failed |
|---|---|---|---|---|
| 1 | 0.099 s | 32.3 tok/s | 32.3 tok/s | 0 |
| 4 | 0.087-0.356 s | 28.5 / 28.8 / 29.1 / 29.4 tok/s | 115.8 tok/s (3.59x) | 0 |

Times are the server's `done` lines (256 tokens each). Peak GPU memory: the WDDM pool went from 2,714 MiB before the
server to 23,315 MiB after load and peaked at 23,324 MiB; the server's working set peaked at 3,973 MiB. The server
warned that `tfgrammar.dll` is missing (a source build). As deployments, not chips: this laptop decoded one stream at
32.3 tok/s against GB10's 32.5, and four at 115.8 against 117.8.
