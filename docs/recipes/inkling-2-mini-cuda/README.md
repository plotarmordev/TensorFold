# Inkling 2 Mini on CUDA: ours against Mia's DGX Spark recipe

Card 257 compares [Mia's AI Lab's recipe](https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold)
(recipe commit `6061fa89a34d`) with TensorFold's own Inkling 2 Mini CUDA engine, natively on the RTX Spark
laptop (Windows 11 ARM64, N1X, `sm_121`). Mia's recipe runs only inside its Linux container on a DGX Spark and
was not run: this page is a feature and matched-harness comparison beside her published figures, not an A/B on
one machine, and every cross-hardware row is labelled non-comparable. The earlier Strata read is
[../inkling-2-mini-strata-study.md](../inkling-2-mini-strata-study.md).

## The two builds

| | Mia's recipe | Ours (this branch) |
|---|---|---|
| Engine base | `tensorfold-native` from PR #4441 at `dc2ca8a1` (upstream main `1587c8caee`) plus `patches/0001`-`0017` | the same #4441 engine, merged as `inkling2.native` (`mini_native.zig`/`mini_model.zig`/`mini_backend.zig`, model type `inkling2_loop`), unpatched |
| Host | NVIDIA PyTorch container, aarch64 Linux, Docker | `tensorfold-native.exe`, Windows ARM64, no container |
| GPU | DGX Spark GB10 (`sm_121`, 273 GB/s LPDDR5x) | RTX Spark N1X (`sm_121`, laptop power cap) |
| Checkpoint | `thinkingmachines/Inkling-2-Mini` pinned `7a67ae23`, SHA-256-checked | the same upstream checkpoint (the laptop's copy is revision `ed40e7da62`) |

Upstream closed #4441 on 23 Nov as superseded — Mini on CUDA goes through #4387's engine there (#4640, then
#4697, #4718, #4728 and #4729 for speed) — while this branch serves the same `mini_*` engine Mia's patches target,
so her series names the same files here but no longer matches their contents. Her published numbers are taken as
her recipe's reference; our side is measured by the two harnesses below.

## Feature table

| Mia's recipe (its defaults) | Ours, merged engine |
|---|---|
| CUDA DSpark drafter (`DRAFTS=1`, patch 0002) | refused by name: `--drafter` exits `NoDraftHead` (C5 pending) |
| fp8 KV cache (`KV_DTYPE=fp8`, patch 0003) | bf16 KV per slot |
| FP8 attention projections (`TF_INKLING2_ATTN_FP8`, patch 0012) | bf16 attention; experts serve as stored (modelopt-nvfp4) |
| Blackwell FP4/MXFP8 prefill tensor cores (`TF_INKLING2_PREFILL_TC`, patch 0015) | the shared widened-code expert path |
| Wide verify CUDA graphs, 4,096-row prompts (patches 0014, 0009) | per-shape graphs, one batched forward a round |
| Native `inkling2_vision` tower (patch 0016) | refused by name: image parts exit `Inkling2MiniImages` (C6 pending) |
| Structured output and `tool_choice` on Mini (patch 0013) | serves; `tfgrammar.dll` beside the exe as on other families |
| `CONTEXT=262144`, `PARALLEL=8`, admission refusal | `--parallel`/`--context` fit admission computes from the pool; the harness serves 8 at 8,192 |
| `Derisked` derivative support, `reasoning_effort` API (patches 0004, 0010) | the checkpoint's own reply parser |

Nothing in her stack is Windows-native: the image, its Zig download and `libtfgrammar.so` are aarch64 Linux,
and no Windows ARM64 Inkling recipe exists in her organization (the RTX PRO 6000 variant targets x86_64 Linux,
`sm_120`). Her `tools/bench_streams.py` also imports a `tools/bench.py` the recipe does not ship, and it sums
per-stream rates where her published sparkDash table divides post-first-token totals by the concurrent wave's
window, so the shipped script is not a runnable reproduction of her published rows; the table stands as
published.

## The pinned common harness

One harness, two front ends, same gates on both machines:

- Contract: `tf-cuda-test kernel-contract "inkling2 loop"`, `inkling2-serving` (#4387's checks), and
  `inkling2-mini-ref` (32 greedy tokens equal to the float64 host reference) plus its planted control, which
  must go red with its own differing tokens.
- Serve: `tensorfold-native serve --backend cuda --parallel 8 --context 8192`, greedy 256-token replies at
  1, 2, 4 and 8 streams, over `bench_concurrent`'s code and chat prompts. Decode tok/s uses sparkDash's own
  definitions: a stream's post-first-token rate and the aggregate over the whole wave's window. Prompt tok/s
  and TTFT are each request's own metrics.
- Output agreement: every concurrent reply's token hash against its solo run (`--alone` on Linux; the reply
  bodies on Windows, where there is no Python).
- Memory: peak GPU pool sampled at 500 ms (WDDM `cuMemGetInfo_v2` pool on the N1X; `nvidia-smi` on GB10), and
  the server process's peak working set.
- Failures: any failed stream or a level under 1.5x one stream's total stops the run.

[`run-script.sh`](run-script.sh) runs it on TOP's GB10 through `swarm gpu-run` (one capped job; the checkpoint
is the Desk's staged `Inkling-2-Mini`). [`n1x-run.ps1`](n1x-run.ps1) runs the same steps on the laptop through
its Codync bot, one `-Step` a run (`fatbins`, `images`, `runner`, `server`, `check`, `control`, `serve`), each
inside 9 minutes; it installs, downloads and deletes nothing outside `zig-out` and `$env:TEMP\inkling2-n1x`,
and its serve client is PowerShell's, timed by the server's `done` lines. The laptop run is dispatched by Capri
only, one at a time; the head and step list below are the brief.

## Receipts

### Ours on TOP's GB10, one capped run

Job `top-7ed89a976853` at `d0a93a07f` (a head that predates `origin/main` `6b6822bbd` but carries card 188's fix
rounds `7a5c0da9f` and `7a0732118`): the build and `kernel-contract "inkling2 loop"` passed (6 kernels, 4 planted
controls red), then `inkling2-serving` stopped the run with `host-reference case=1 mismatch=MiniServingTokenMismatch`:
the sampled case (seed 37, temperature 0.8, top_k 64, top_p 0.95) draws other tokens on CUDA than the host does over
the same logits. Greedy (case 0) passed. That failure is recorded at `d0a93a07f`, not yet shown on main, and belongs
to card 263 (or 188, whichever proves it); this card neither fixes nor skips the check. Both runners now print its
exit, go on to the reference, control and serve steps, and return it last, so the job stays red while it measures.
Output agreement below is therefore greedy only; sampled agreement is blocked on 263. The prior receipt of the same
engine and checkpoint (card 211's, job `top-0b05213c9f06` at `4a5089e70`, `--parallel 4`): 32.9 tok/s at one stream,
117.9/118.8 total at four, 0 failed, 0 of 32 reference tokens off, 67% of the 49 tok/s weight-read ceiling (5.55 GB a
token at 273 GB/s).

### Ours on the RTX Spark N1X, native Windows

Pending the laptop dispatch; card 243's runs of this harness's predecessor give the expected order: run 6 at
`2c9228a8` (request relayed by Capri, driver 621.01, Balanced plan) built clean, passed every check and its
control, and served 32.3 tok/s at one stream and 115.8 total at four (3.59x), first token 0.099 s, WDDM pool
peak 23,324 MiB of 51,816 MiB. As deployments, not chips: that laptop run decoded one stream at 32.3 tok/s
against GB10's 32.5, four at 115.8 against 117.8. `docs/WINDOWS.md` records the N1X's 42-49 W cap.

### Against Mia's published DGX figures — not comparable, side by side

Mia's rows are hers: sparkDash on a DGX Spark, `CONTEXT=262144`, `PARALLEL=8`, fp8 KV, DSpark drafts on, FP8
attention and prefill tensor cores on, temp 0.7, top_k 20, thinking on. Ours are greedy, plain decode, bf16 KV,
an 8,192-token window, laptop power cap. Nothing below is a like-for-like ratio.

| Aggregate decode, tok/s | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Mia, prose (published DGX) | 48.3 | 74.9 | 106.1 | 145.0 |
| Mia, code (published DGX) | 56.1 | 86.5 | 108.5 | 149.4 |
| Ours, N1X Windows (card 243 run 6, 4-slot build) | 32.3 | - | 115.8 | - |
| Ours, GB10 (card 211) | 32.9 | - | 118.8 | - |

| Prefill (Mia's published rows) | tok/s | TTFT |
|---|---|---|
| 4k | 3,497.1 | 1.34 s |
| 8k | 3,343.6 | 2.62 s |
| 16k | 3,252.5 | 5.21 s |
| 32k | 3,072.2 | 10.84 s |
| 64k | 2,749.6 | 24.01 s |
| 128k | 2,267.3 | 57.99 s |

Ours has no published prefill row at these lengths; the harness's prefill probe fills this once the laptop run
and the TOP receipt land.

## What the gap is made of

- Drafting. Her rows are drafted; ours has no Mini CUDA drafter, so solo decode is the draft-off arm she never
  published. The 1-stream rows (48-56 drafted against ~33 plain) bound her whole recipe's gain, not the
  drafter's alone: fp8 KV and attention and the fp4 prefill tensor cores are also on in her runs.
- Streams 4 and 8. Her per-stream rate halves from 1 to 4 streams (37.6 -> 27.9 prose); ours loses about 10%
  (32.9 -> 29.7). Batched drafting in her patch 0006 keeps her aggregate ahead only through 8 streams; the
  crossover point is one of the things the laptop run at this head measures.
- Windows natively. A Windows-native N1X run of her stack does not exist, and the two transferable units of
  work are already enumerated in the Strata study's follow-ups (a CUDA DSpark tree and fp4 grouped experts at
  M >= 24); both are upstream-engine changes, not this card's doc scope.

## Laptop-ready brief

For Capri's next dispatch, at this branch's head:

1. `git fetch` the card branch (approved fetch only; the checkpoint `C:\models\Inkling-2-Mini-ed40e7da62` is
   already staged and is not to be touched).
2. `-Step fatbins`, `-Step images`, `-Step runner`, `-Step server`: the four build steps, each under 9 min,
   `-Dsm=121 -Doptimize=ReleaseFast`, CUDA 13.5 `windows-arm64` nvcc and `cl.exe` arm64.
3. `-Step check`: `info`, `kernel-contract "inkling2 loop"`, `inkling2-serving`, `inkling2-mini-ref`.
4. `-Step control`: the planted control must exit non-zero with its differing tokens.
5. `-Step serve`: ready wait under 240 s, prefill-short and prefill-long probes, then 1, 2, 4 and 8 streams,
   output-agreement counts, WDDM pool peak at 500 ms sampling, server peak working set, all inside 9 min.

Every `RESULT n1x` line and each step's exit code is the receipt. Exact Windows errors go back to Capri.
