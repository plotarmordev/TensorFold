# Windows CUDA build with MSVC-hosted nvcc (sm_121)

Build-only receipt for card 213 at `bccab3ab0763dcfc8ec91efaab7f61095352b6a2`, relayed by the Desk on 2026-11-24
(request `bdb8d0a2`). No tests, serve or benchmark ran; the GPU suites are in [WINDOWS.md](../../WINDOWS.md).

- Host: the RTX laptop (Windows on Arm64). MSVC 14.44.35207 HostARM64 `cl.exe` (`vcvarsall arm64`, exit 0),
  CUDA 13.5 (`nvcc` V13.5.37), Zig 0.17.0.
- Clean checkout: `git fetch`, `git checkout --detach bccab3ab`, `git clean -fdx` all exit 0; `git status --porcelain`
  empty before and after the build. No patch applied.
- Command, 06:24:16 to 06:34:02 local (9 min 46 s):

  ```bat
  zig build native install -Dsm=121 -Dstrip=true -Dnvcc=%CUDA_PATH%\bin\nvcc.exe -Doptimize=ReleaseFast --summary all
  ```

- Result: `EXIT=0`, `Build Summary: 230/230 steps succeeded`; all 19 installs and the `native` step succeeded.

| `zig-out\bin\` | Bytes |
| --- | ---: |
| `tf-cuda-test.exe` | 108,113,408 |
| `tensorfold-native.exe` | 57,839,104 |
| `tf-muse.exe` | 9,387,520 |

The full log (`build213_bccab.log`, 4,368,099 bytes, 17,179 lines) stays on the laptop. It has no line containing
`error`. Its 13,764 warning lines are all MSVC C4819 (a character outside code page 936, the laptop's GBK locale):
13,728 in `cuda_runtime_api.h`, 12 in `cuda.h`, and 8 on line 1 of `zig/kernels/cuda/h3_int8_tma.cuh`, `h3.cu` and
`h3_image_prompt.cu`. Those warnings fill the 176 nvcc stderr blocks (139 under `tensorfold-native`, 37 under
`tf-cuda-test`); the other lines are the build summary tree.
