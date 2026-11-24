# TensorFold on Windows (native)

`tensorfold-native.exe` runs on Windows with an NVIDIA GPU through the same CUDA engine the Linux build uses, and with an
AMD GPU through the HIP backend when the build opts in (`-Dhip=true`). Metal is macOS's. Windows is supported from
source only.
The PR reports RTX 4090 (`x86_64-windows`, `sm_89`) and NVIDIA RTX Spark N1X (`aarch64-windows`, `sm_121`) runs.
Those device receipts have not been reproduced on the maintained tree.

## What you need

- An NVIDIA GPU and a current display driver (it ships `nvcuda.dll`; no CUDA toolkit is needed to run).
- To build: [Zig 0.17.0](https://ziglang.org/download/) (the version in `.zig-version`), the CUDA Toolkit
  (13.x tested) for `nvcc`, and MSVC C++ Build Tools with the C++ workload (`cl.exe`, nvcc's host compiler).

## Build

From a "x64 Native Tools" prompt (or after running `vcvars64.bat`, or `vcvarsall.bat arm64_amd64` on Windows ARM64), so
`cl.exe` is on `PATH`:

```bat
zig build native install -Dsm=89 -Dstrip=true ^
  -Dnvcc="%CUDA_PATH%\bin\nvcc.exe"
```

`-Dsm` takes the GPU's compute capability without the dot, comma separated for several: 86 (RTX 30), 89 (RTX 40),
120 (RTX 50), 121 (GB10 / RTX Spark N1X). The server is `zig-out\native\bin\tensorfold-native.exe`;
`zig-out\bin\tf-cuda-test.exe` runs the GPU suites (`tf-cuda-test info`, `smoke`, `glue`, `sample`, `quant-ref`,
`affine8`, `modelopt`, `exlr8`).

`zig build test` runs the host tests without a GPU.

### The host tests on a machine with no usable GPU

The host suite needs only Zig: no CUDA toolkit, no MSVC (Zig's own clang builds the C++ host files), nothing elevated.
Run on a 2015 Windows 10 laptop (Intel i7-4720HQ, four cores, 16 GB; its GTX 965M is below the CUDA build's sm_80
floor) with Zig 0.17.0 from the portable zip, on main at 9d6f1d78c (2026-11-04):

- `zig build test` from nothing: exit 0 in 1607 s. Then, with `.zig-cache` removed and the global Zig cache warm,
  `zig build test --summary all`: `Build Summary: 491/491 steps succeeded; 6361/6514 tests passed (153 skipped)`, exit
  0, 1119 s. It prints little for minutes at a time; the summary line and the exit code are the result.
- Zig 0.17's build runner prints a step's tree, its captured stderr and a `failed command:` line for every step that
  wrote to stderr, warnings included (the simulated device-loss warnings, numeric tables, the `tfgrammar.dll` line
  below). A failed step has an `error:` line and the build exits non-zero; neither happened in that run.
- The run named 118 distinct skipped tests behind its 153 skips: 37 behind `os.tag == .windows` in the test's own
  source (the key file's 0600 permission tests, request_log, drama_voice_tool, media_fetch_tls and CUDA tp_watch among
  them), 9 behind "symlinks need privilege on Windows", 22 macOS-only, 9 Linux-only, and 41 behind a checkpoint,
  fixture, device or another guard. Developer Mode (Settings, For developers) or an elevated prompt runs the nine
  symlink tests, which skip only on `PermissionDenied`; three more symlink tests, in `core/learned_part_test.zig`,
  skip on Windows whatever the privilege (an `os.tag == .windows` guard) and are among the 13 skip events the
  one-by-one run did not name.
- Unzip the Zig download with Windows' own `tar.exe` (`C:\Windows\System32\tar.exe -xf zig-x86_64-windows-0.17.0.zip`,
  37 s); PowerShell's `Expand-Archive` took 258 s, and Git for Windows' GNU `tar`, when its `usr\bin` is first on
  `PATH`, does not read zips.
- A test on a from-source tree logs a warning, `tfgrammar.dll is required for structured output and required tool
  calls`; see [Structured output and required tool calls](#structured-output-and-required-tool-calls) for where a
  source build gets it.
- The `python3` the tools' commands name may be the Microsoft Store alias on Windows; a python.org install is `python`.

### HIP

Add `-Dhip=true -Dhip-arch=<arch>` and name a HIP compiler; the build takes `hipcc.exe` first and `hipcc.bat` second,
from `-Dhipcc`, `PATH`, or `HIP_PATH`/`ROCM_PATH`'s `bin`, and runs a `.bat` through `cmd /c call`:

```bat
zig build native-hip -Dhip=true -Dhip-arch=gfx1151 ^
  -Dhipcc=C:\therock\gfx1151-10.1.0\bin\hipcc.exe
```

TheRock's Windows `gfx1151` dist and a HIP SDK's `hipcc.exe` both build the kernels. The server loads the runtime from
`amdhip64.dll` (the driver's copy in System32), then from a HIP SDK's `amdhip64_7.dll` and `amdhip64_6.dll` on `PATH`;
`TENSORFOLD_HIP` names one library outright, on every platform. Admission takes its budget from the driver, and the
optional libraries are found by their Windows names (`rocblas.dll`, `MIOpen.dll`, `cublas64_13.dll`).

The Windows HIP SDK does not implement cooperative launches (ROCm/HIP#3640), so a grid that asks for one runs plain; the
engine's own launches do not use them. Without a GPU, `zig build hip-host-test -Dhip=true` runs the admission tests and
`zig build check-native-hip-host -Dhip=true` compiles the real server against inert kernels. Tested on `gfx1151`
(Ryzen AI MAX+ 395): `hip-gpu-test` 38/38 on the device, and `capabilities --json` reports the HIP backend with rocBLAS
and MIOpen found.

## Run

```bat
tensorfold-native pull TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
tensorfold-native models
tensorfold-native serve <model path> --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

The server listens on `http://127.0.0.1:8080/v1`. Ctrl+C (or closing the console) stops it cleanly.

### Structured output and required tool calls

`response_format` and a required or named `tool_choice` need the grammar library, `tfgrammar.dll`, in
`zig-out\native\lib` beside the server's `bin`. Without it those requests get HTTP 400 and everything else serves.
Build it from the pinned xgrammar checkout:

```bat
git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar
zig build grammar -Dxgrammar=%CD%\xgrammar
```

Tested on an RTX 5090 with Nemotron 3.5 Lightning: OpenAI `"required"` and named tools, Anthropic `any` and `tool`,
`json_schema` and `json_object`.

## Models on CUDA

Nemotron 3.5 Lightning serves as on Linux. The Qwen3.5 hybrids have a CUDA engine of their own in this port
(`zig/src/families/qwen3_5/cuda*.zig`, kernels `zig/kernels/cuda/qwen35*.cu`), the Metal kernels' arithmetic with
the same rounding points, so a window of up to 16 rows reproduces one-token steps bit for bit:

```bat
tensorfold-native serve <Qwen3.5-2B-MLX-4bit> --name qwen2b --context 32768
tensorfold-native serve <Qwen3.8-27B-MLX-4bit> --name qwen27b --context 8192 --parallel 4 ^
  --drafter <Qwen3.8-27B-DFlash2>
```

- Qwen3.5-2B: about 340 tokens/s for one request on an RTX 4090, 1,500 across eight.
- Qwen3.8-27B (16 GiB): 40 tokens/s plain, at the card's memory bandwidth; 240 across eight requests.
- `--drafter` (4-bit only) loads z-lab's DFlash2 draft model: the target writes five layers' residuals into a ring
  for it, and it drafts a tree of up to 12 lanes a round that the round loop's planner shapes from what the target
  took. A lone greedy request decodes at 80 to 160 tokens/s by how predictable the text is (prose at the low end,
  code at the high), sampled ones at 50 to 70, and four at once reach about 160 aggregate; replies stay identical to
  plain decoding. Loading takes about 80 s: the drafter's weights are quantized to 4 bits on the host and every
  shape is run once, as the driver stalls a process's first use of each kernel and buffer.
- A verify window's rows are not snapshotted: each stream logs its rows' recurrence updates and a keep replays the
  accepted path, so the 27B records 32 rows a forward where 8 fit before.
- `TF_TIMING=1` prints each verify, keep, absorb and hold's milliseconds and the drafted shape (chain or lanes);
  `TF_NO_TREES=1` drafts chains only.
- No prompt cache (`--prompt-cache-gib`), `--learn` or `--slide` for these two models.

### Mixed-width checkpoints

The Qwen3.5 engine also loads MLX checkpoints whose modules carry their own widths (the config's `quantization`
table names a width per module, as mlx-lm's mixed recipes and the OptiQ and oQ conversions write), 2 to 8 bits in
groups of 64. Each module stays at its own width: 4-bit modules on the 4-bit tiles as before, 2, 3, 5 and 6-bit ones
on `affine8.cu`'s tiles of that width, and 8-bit ones on its byte tiles (a 7-bit module would take the 8-bit tiles).
Every width runs the 4-bit lane GEMV's arithmetic and hands its MMAs the same bf16 operands, so a code gives the
same bits whatever width it is stored at, and nothing is rounded. An embedding of a width other than 4 or 8 is
widened to the next at load. Windows still reproduce one-token steps bit for bit, and a uniform 4-bit checkpoint
loads and runs exactly as before.

- Tested on an RTX 4080 SUPER (16 GB): `mlx-community/Qwen3.5-4B-OptiQ-4bit` (4 and 8 bits, a tied 8-bit
  embedding) at about 130 tokens/s for one request and 400 across four, and `mlx-community/Qwen3.5-9B-OptiQ-4bit`
  (an untied 8-bit head) at about 78 and 260; replies alone, at four streams and with `"draft": false` equal.
- `TensorFold/Qwen3.8-27B-oQ2` (mostly 2-bit, 136 modules at 5 bits) takes 10.3 GiB, so the 27B fits a 16 GB card:
  one stream at a 32k window, about 45 to 64 tokens/s with drafts. The card itself calls oQ2 an aggressive
  low-bit release, and its replies show it.
- Prompt chunks take a GEMM over the same tiles (`affine_prefill.cu`) with the 4-bit prompt GEMM's arithmetic (each weight rounded once to
  bf16, one fp32 chain over K), so a code of up to 4 bits gives the bits the 4-bit prompt GEMM gives it. On the
  9B 8-bit it lifts prompts from about 1,500 to 1,900 tokens/s; the 27B oQ2's prompts (about 770 tokens/s) were
  already bound by the card's bf16 tensor rate with fp32 accumulation, not by weight reads.
- `tf-cuda-test affine8` checks each narrow width's products against its codes widened onto the 8-bit tiles, bit
  for bit, the products against float64, every row count and split-K path byte for byte, and the prompt GEMM
  against the 4-bit and 8-bit tiles' prompt GEMMs.

## How the port works

- `zig/src/platform/sys.zig` (`@import("sys")`, wired into every module by `zig/build/sys.zig`) is the operating
  system as the engine calls it. On Linux and macOS it is `std.c`, `std.posix` and `std.DynLib` unchanged; on Windows
  `win_c.zig` and `win_posix.zig` give the same names over the UCRT, Win32 and Winsock.
- Files are opened with every share mode, so rename-over and unlink-while-open behave as on POSIX. Checkpoint reads
  skip the page cache (`FILE_FLAG_NO_BUFFERING`, as `O_DIRECT` on Linux). `/tmp` is the user's temporary folder and
  `HOME` falls back to `USERPROFILE`.
- The Hugging Face cache links snapshot files to blobs. Windows allows symlinks only with Developer Mode or elevation,
  so `pull` falls back to hard links (no extra space). `tensorfold rm` handles those hard links, and the copies
  huggingface_hub makes without links, the same way as links, and counts each file's bytes once.
- Checkpoint reads use threads (io_uring is Linux's).
- Integrated unified-memory GPUs on Windows (such as RTX Spark N1X `sm_121`) admit against the WDDM CUDA device pool
  (`cuMemGetInfo_v2`) while keeping the unified 10% reserve and 4 GiB default floor. A fresh device receipt must check
  the WDDM budget, allocation deltas and equality cells; source admission alone does not qualify the device.
- Under WSL2 the Linux build sees the same integrated N1X, but `/proc/meminfo` there is WSL's RAM (about 30 GiB on a
  128 GB machine whose firmware gives the GPU 96 GB), not the GPU's pool. With `WSL_DISTRO_NAME` set the integrated
  admission reads the CUDA pool, as on Windows; `TENSORFOLD_UNIFIED_HOST_COUNTS=1` restores procfs, `=0` forces the pool,
  and the CUDA backend refuses any other value at startup, naming the variable.
- The API key file is checked for type and size; Windows keeps access in ACLs, not mode bits, so the "unreadable by
  other users" check is the folder's ACL rather than `chmod 600`.

## Differences from Linux

- No io_uring, `ulimit` or Linux `/proc/meminfo` probe, and no `SIGHUP` (the key file is still reread when it changes).
- `TF_HEAT_*` thermal pacing reads Linux's `/sys/class/thermal` and does nothing on Windows.

## RTX Spark N1X: a receipt (2026-11-04)

An HP OmniBook X 14 (RTX Spark N1X, 128 GB, driver 617.21 / CUDA 13.4), HP Performance mode, plugged in, 1.2.2 with the
WSL admission change above:

- Native build: Zig 0.17.0, the CUDA 13.4 Windows-on-Arm toolkit assembled from NVIDIA's redistributable archives
  (`redistrib_13.4.3.json`, `windows-arm64`: nvcc, cudart, crt, cccl, nvvm, nvrtc, nvJitLink, nvptxcompiler, cuBLAS,
  NVTX, profiler API; winget's `Nvidia.CUDA` has no arm64 installer), VS 2022 Build Tools with the ARM64 C++ tools
  (`vcvarsall arm64`), `zig build native install -Dsm=121 -Dstrip=true`. `tf-cuda-test info`: sm_121, 48 SMs,
  113,103 MiB total; `smoke` passes; `glue` 1473 of 1473 byte-equal.
- `Blackfrost-AI/CYBER-FROST-3.8-FP4-Local` (Flash Next, ModelOpt mixed): 72.65 GiB, 4 streams at a 32,768-token
  window, ready in 35-46 s. vcruz305's `bench_v1.py` / `concurrency.py` clients, greedy, 400 tokens, client end-to-end:

| | WSL2 (Linux build) | native Windows |
| --- | ---: | ---: |
| One stream, code / DevOps / prose (tok/s) | 43.8 / 42.2 / 35.7 | 50.4 / 40.2 / 37.3 |
| First token, short prompt (s) | 0.21-0.24 | 0.22-0.28 (clients over a ~6 ms tunnel) |
| First token, 17,850-token prompt (s) | 10.2 | 10.9 |
| 1 / 2 / 4 streams, aggregate (tok/s) | 36.4 / 50.2 / 65.6 | 36.6 / 54.4 / 70.0 |
| Left for streams after the model and reserve | 15.4 GiB | 19.0 GiB |

The GPU ran at about 2.1 GHz and 42-49 W (the laptop's power cap; in Windows' power-saving mode it runs near 1.4 GHz).
The same laptop serving the same model through exllamav3 (vcruz305's EXL3 recipe, Cyber Frost 3.87 bpw) gave 38.6 /
34.5 / 27.8 tok/s, a first token in 0.74-0.80 s, and 36.2 tok/s aggregate at four streams.

## RTX Spark N1X: what the 1.3.3 build lacked, and what Inkling 2 Mini still needs (2026-11-22)

A second N1X laptop (Windows 11 ARM64, driver 621.01 / CUDA 13.5, 48 SMs, 51,816 MiB of GPU memory as its firmware
shares it) with Zig 0.17.0 and the CUDA 13.5.0 `windows-arm64` redistributables failed
`zig build native install -Dsm=121 -Dstrip=true` at `1f68cdb4b`. Fixed since:

- 11 errors in `tf-muse` (`std.c` has no `pollfd` in `ws2_32`, `void` parameters under `aarch64_aapcs_win`, POSIX
  `pthread_mutex_t`, `MSG.NOSIGNAL` and null-handle types in `zig/src/fabric/tp`): its rank 1 follows over the POSIX TP
  session, so Windows builds skip it, as they skip the TP tests.
- `tensorfold-native`: `gpu_turn.zig` set POSIX file modes (`Permissions.fromMode`), and DXGI's COM calls cast untyped
  vtable entries to aarch64 function pointers without `@alignCast`.
- `zig build compile-native-cuda-windows` now type-checks the `aarch64-windows-gnu` sm_121 product too.
- `glm_index_rope.cuh` marked `turn` GCC's way (`__attribute__((noinline))`), which nvcc with cl.exe rejects, so
  `glm_glue` and `glm_index` did not build; it is `__noinline__` now. `compile-native-cuda-windows` embeds empty
  images and cannot catch a kernel source that only cl.exe rejects.
- Windows fatbins pass `-Xcompiler=/wd4819`, the flag that showed that error under the laptop's code page 936, where
  C4819 warnings from the CUDA headers buried it. The build still prints C4819 warnings with it; which nvcc pass
  they come from is not known.

Then, from a clean checkout of `8365021a7` on the same laptop, run by its only operator, the Codync bot
`codync@rtx-spark` (git 2.56.1, Zig 0.17.0, nvcc 13.5.37 with `cl` 14.44.35207 arm64 from VS 2022 Build Tools 17.14,
`vcvarsall arm64`; no Python or WSL; power mode not recorded):

- `zig build native install -Dsm=121 -Dstrip=true -Dnvcc=%CUDA_PATH%\bin\nvcc.exe` exits 0 in 629 s, 222 of 222
  steps.
- `tf-cuda-test` (sm_121, 48 SMs, 51,816 MiB), passed / failed, 29/0 in all: `smoke` 8/0, `glue` 1/0 (1473 of 1473
  byte-equal), `sample` 2/0 (2880 draws, none differ), `quant-ref` 4/0 (worst 0.510 of its tolerance), `affine8` 1/0
  (worst 0.338), `modelopt` 6/0, `exlr8` 7/0 (worst 0.1084).
- `zig build kernel-contract -Dsm=121 -Dnvcc=%CUDA_PATH%\bin\nvcc.exe` ended with exit code 5, which the program
  never returns, and no SUMMARY line: 525 PASS, 47 planted controls red, 26 N/A and 12 FAIL, every FAIL in
  MiniMax-H3's image kernels. None of the 12 was Windows', and the exit was the test's heap:
  - Gate 1 grows `h3p_scatter_add` and `h3p_splice`'s 4-row shape to 5 rows, and their indices `2i` wrote past the
    8-row buffer (TOP's Linux run at `137959df8` died after `h3p_add` with "double free or corruption"); they now
    spread over its rows.
  - Nine `shapes` and `sums` lines (`h3d_embed`, `h3d_rms_norm`, `h3d_layer_norm`, `h3d_attention`, `h3p_norm`,
    `h3p_heads` rotary, at 1.01 to 1.81 of their bound) were the references: six ended in a bf16 rounding of their
    own, so one tie the kernel's fp32 rounded the other way put them a whole output step away, where the bound
    allows half. They end unrounded now, `h3p_norm` qwen's throughout, and `h3p_heads` rotary declares its four bf16
    points. Linux sm_121 on TOP gave the same lines and ratios, and with these changes
    [`contract-job.sh`](recipes/minimax-h3-cuda/contract-job.sh) there (`top-b6b76efdc239` at `ba92e161f`) gives "26
    kernels: 26 pass gates 1-4, 0 fail, 0 known findings; 0 limit problems", exit 0. That is Linux's result, not
    this laptop's.
  - Two `h3p_attention` `sums` lines "refused a served shape" on any OS: gate 4 cuts the 77-row shape to 4 rows and
    `prepare` refused fewer rows than K. The kernel's keys are its rows, so `prepare` no longer reads K.
  - `h3p_norm`'s 1024 static shared bytes (recorded 4) come from the `__align__(1024)` shared array
    `h3_int8_tma.cuh` adds to its image; the records say 1024. MLP128's two N/A lines now name its sm_120-only tile.

The same laptop reran a clean checkout of `3332b7305` detached (2026-11-23), again through `codync@rtx-spark`:

- `zig build native install -Dsm=121 -Dstrip=true` exits 0 in 530 s.
- `tf-cuda-test`: all eight suites exit 0 (`info`: CUDA 13050, sm_121, 48 SMs, 51,816 MiB; `smoke` 8/8; `glue` 1473
  of 1473 byte-equal; `sample` 2/2, 2880 draws, none differ).
- `zig build kernel-contract -Dsm=121` exits 1, 185 of 187 build steps, with no overall SUMMARY line. Gates 1 to 4
  and the standalone suites ran to their end (the last of them printed "SUMMARY prompt_swiglu: 3 pass, 0 fail; 2
  controls red, 0 missed"); gate 5 (`limits live`) stopped at its first failed lookup, so the launches after it,
  the gate 1-4 tally and the limit count have no result:

  ```text
  error: cuModuleGetFunction: CUDA_ERROR_NOT_FOUND (500) named symbol not found
  error: kernel symbol not in module: _ZN6tfl2pf15segments_kernelILb1EEEvPKlij
  FAIL kernel-contract: NotFound
  ```

- 985 FAIL lines: 281 planted controls, red as planted, and 704 on the laptop's sm_121 Windows build: 471 limits
  (470 `limits live`), 74 nonfinite, 65 shapes, 62 repeat, 30 sums, and the abort's two. The receipt counts them by
  gate; a count by family needs the laptop's log.
- MiniMax-H3: 179 PASS and one numeric FAIL, the 12 lines of `8365021a7` gone: "h3m_sinusoid: at 8x256x1 output 8
  is -2.091842e-1 where float64 gives -2.091841e-1: 1.25 of its bound". `3332b7305` does not hold main's later
  bound for that kernel (`c6557d036`, the f32 angle's error), so that bound is untested on Windows.

A third clean checkout, `2d0a39ce3` (main merged in, with `c6557d036`'s `h3m_sinusoid` bound), ran there the same day.
These figures are the operator's receipt; its logs stay on the laptop (`workspace\scripts\r2d0a\`):

- `zig build native install -Dsm=121 -Dstrip=true` exits 1, 225 of 228 steps. Only `tf-qwen3-asr` fails: its timer
  called POSIX `clock_gettime` (`std/c.zig`, "parameter of type void not allowed ... aarch64_aapcs_win").
  `compile-native-cuda-windows` checks the server alone, so no host caught it. `tf-cuda-test.exe` was built.
- `tf-cuda-test`: all eight suites exit 0 (`glue` 1473 of 1473 byte-equal; `sample` 2880 draws, none differ).
- `zig build kernel-contract -Dsm=121` exits 1 after 2 h 40 min on one CPU core, with up to 18 minutes between
  lines, and reaches its SUMMARY: "844 kernels: 841 pass gates 1-4, 3 fail, 11 known findings; 500 controls: 500 red
  as planted, 0 missed; 1013 limit problems".

Since that run:

- `tf-qwen3-asr` reads the awake clock through `std.Io`, and builds for `aarch64-windows-gnu`.
- `d3_cfg_mix` "refused a served shape" on every OS: gate 1 grows its one row to five and the test refused any row
  count but one. The mix is one value per logit, so its rows are more logits through the same launch.
- `kernel-contract` prints SUMMARY when a suite or gate 5 stops it too, marked "(stopped early)", and a COUNT line
  after each kernel. `zig build kernel-contract -Dcontract-only=WORD` (repeatable) passes the words
  `tf-cuda-test kernel-contract WORD...` already took: the kernels and launches so named, without the suites.
- `tf-cuda-test kernel-contract I/N [WORD...]` runs every N-th of those kernels and launches, so N processes started
  side by side cover a run once on N cores. Each prints its own COUNT lines and a SUMMARY marked "(shard I/N)"; the
  first also runs what is not divided (the suites and the offline audit, on a run with no words). The run passes
  when every shard exits 0. `zig build` runs its Run steps one at a time, so the caller starts the processes
  (`docs/recipes/minimax-h3-cuda/cuda-build.sh` does, for `H3_SHARDS`).

On TOP's GB10 (sm_121, Linux, nvcc 13.0) at `03e9b18df`, `kernel-contract h3d h3p d3_cfg_mix` passes gates 1-4 on
all 27 kernels: `d3_cfg_mix` "3 non-finite and 20482 finite outputs match the float64 classes", with a COUNT line
per kernel and the SUMMARY. It exits 1 on one limit problem, `d3_cfg_mix` "12 registers ... (recorded 0, 0)": that
record is `registry.none`, pending on every OS. At `a29b38dcd`, `contract-job.sh` (`h3d h3p tf_me_conv
tf_idol_dec_attention`, four shards) exits 0: 28 kernels pass, the planted control red, no limit problem.

The laptop's rerun 11 at `a29b38dcd` (Desk request `6203c38d`) qualifies Windows: the clean build and all eight
`tf-cuda-test` suites pass, and `kernel-contract -Dsm=121` reaches its SUMMARY with 844 of 844 kernels passing gates
1-4, 500 of 500 controls red as planted and 11 known findings. It exits 1 only on 1013 limit problems, mostly
"(recorded 0, 0)": there are no records for Windows nvcc 13.5 sm_121 images (item 7). The two contract corrections
since (`6b692d6ee`, `41b7b7004`) touch neither the build nor the suites; their laptop re-check at `75d7bdcd7` is
queued.

This card neither downloads nor serves a model on the laptop. Its one CUDA serve there is card 243's Inkling 2 Mini
receipt (`docs/recipes/inkling-2-mini-cuda/README.md` on `feat/243-inkling-2-mini-runs-on-net-s-rtx-spark-l`): run 2 at
`2a37918af`, a clean clone, `n1x-run.ps1 -Step build`, `-Step check` and `-Step serve` all exit 0; 32 of 32 greedy
tokens equal the float64 reference; 256-token greedy replies decode at 32.2 tok/s on one stream and 117.0 in total on
four, first token 0.104 s.

On this branch Inkling 2 Mini is not served on any OS: `zig/src/families/inkling2` has kernels, ops and the host-side config,
names, plan and prompt, which `tf-cuda-test`'s contracts use (they compile for `aarch64-windows-gnu`). For Windows it
needs:

1. A native family (`open`, weights, forward, admission) registered in `zig/build/cuda_engines.zig` and
   `zig/src/native/cuda_families.zig`, so `compile-native-cuda-windows` checks it for both Windows targets.
2. No serving path through `cuda.tp_watch`, `cuda.tp_group` or `cuda.carveout` (sockets, `fork`, DRM `ioctl`), which
   do not compile for Windows.
3. `test-inkling2-host` built for `b.graph.host`, not `-Dtarget` (`zig/build/inkling2_host.zig`), as the other host
   packages are, so a cross-target `zig build test` does not try to run a Windows binary.
4. Admission against the WDDM pool: the 20.2 GB checkpoint leaves about 20 GiB of this laptop's 40 GiB CUDA share for
   the cache and streams.
5. Its CUDA sources in CUDA's qualifiers (`__noinline__`, `__forceinline__`), not GCC attributes: only a build with
   nvcc and cl.exe, such as this laptop's, rejects them.
6. Portable kernel names. The first missing symbol, `tfl2pf::segments_kernel<true>`
   (`_ZN6tfl2pf15segments_kernelILb1EEEvPKlij`), is compiled from `const int64_t*`: LP64 encodes `long` as `l`, while
   Windows LLP64 encodes its `long long` as `x`. The affected `l2pf`, `l2pace`, DeepSeek `x3ld`/`x3pf`,
   `tf_exl3x::grouped_kernel` and two unpack-kernel families need portable names before their Windows serving paths
   qualify. Confirm the actual Windows module symbols from its fatbin. Gate 5 now counts each missing function as one
   FAIL and continues to SUMMARY instead of aborting at the first lookup.
7. Gate 5 records for sm_121 on Windows. `2d0a39ce3` gave 1013 limit problems (`3332b7305`: 470 `limits live` lines).
   A launch counts when it does not fit the driver's threads or shared bytes for that function, or when its
   registers or static shared bytes differ from its record; the records are nvcc 13.3's and the laptop's images are
   nvcc 13.5 with cl.exe. The receipt's example, "heavy at 1,024 threads: 1024 launched, driver allows 384; 166
   registers", is the planted control `tf_ctl_heavy`, red as planted and not counted. Which of the 1013 are launches
   that do not fit and which are stale records takes the laptop's log: fix the first, and give the second a record
   keyed to that nvcc and host target from the run's own figures. Do not copy older Linux figures or guess values.
   Two parts of the 1013 need no laptop: the offline audit, which a full run adds on any OS, counts 203 problems
   over the 1755 records at this branch's head, and 172 of the 1748 launches live on sm_121 have no recorded
   registers for an nvcc 13.5 image (99 for 13.3, 7 for 13.0), which gate 5 counts as it did `d3_cfg_mix` on TOP.
8. Three of 844 kernels failed gates 1-4 at `2d0a39ce3`. `d3_cfg_mix` refused gate 1's five-row shape;
   that fixture now accepts more rows as more logits. At `03e9b18df`, the laptop's next full run reported two
   remaining gate failures: `tf_me_conv` left its signed interval at gate 4 because fp32 `expf(-s)` overflows
   for a stress sum of -89, yielding -0 instead of float64 silu's tiny negative result; the reference now
   includes zero in that interval. `tf_idol_dec_attention` exceeded its shapes bound by 1.07 at 16x1000x65:
   the reference rounded P at the final row max, while the tiled kernel rounds against the highest score of every
   tile so far before rescaling. Its reference now carries that running max across tiles. With both corrections
   the laptop's rerun 11 passes all 844 kernels (above).
   The 11 other non-control FAIL lines are declared known findings: qwen35 decode attention can leave a
   NaN-key row finite, and `tf_glm_scores` maps a NaN score to zero via `fmaxf`. Those served-kernel
   behaviors are unchanged by this card.
9. A full contract inside a laptop window. It is one thread and one CUDA stream a process, so shards (above) are
   how it uses more cores, but the words and the shard filter only after every kernel's fixture is built: each
   process pays that build and holds its GPU memory, which N processes multiply. Run one family at a time
   (`kernel-contract I/N WORD`), and raise N only while the 40 GiB CUDA share holds.
