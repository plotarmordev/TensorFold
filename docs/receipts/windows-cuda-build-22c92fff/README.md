# Windows CUDA build with MSVC-hosted nvcc (sm_121)

Build-only receipt for card 213 at the merged head `22c92fff14f9fb2abe5b9e8a416c879ffdc88225`, run on the laptop
on 2026-11-24 and relayed by Capri on 2026-11-25. No tests, serve or benchmark ran; the GPU suites are in
[WINDOWS.md](../../WINDOWS.md).

- Host: the RTX laptop (Windows on Arm64). VS 2022 Build Tools 17.14.42, MSVC 14.44.35207 HostARM64 `cl.exe`
  (`vcvarsall arm64`, exit 0), CUDA 13.5 (`nvcc` V13.5.37, `cuda_13.5.r13.5/compiler.39125733_0`), Zig 0.17.0.
- Clean checkout (`workspace\tf213`): `git status --porcelain` empty, no patch applied.
- Command, 14:50:51 to 15:00:57 local (10 min 6 s):

  ```bat
  zig build native install -Dsm=121 -Dstrip=true -Dnvcc=%CUDA_PATH%\bin\nvcc.exe -Doptimize=ReleaseFast --summary all
  ```

- Result: `EXIT=0`, `Build Summary: 230/230 steps succeeded`, which include `tf-cuda-test.exe` (108,119,552 bytes),
  `tensorfold-native.exe` (`compile exe tensorfold-native safe native success 3m MaxRSS:3G`) and `tf-muse.exe`.

## Build log

The collapsed log is committed beside this README; the full log stays on the laptop. These hashes identify both:

| File | Bytes | Lines | SHA-256 |
| --- | ---: | ---: | --- |
| `build213_22c9.log` (full) | 4,367,960 | 17,176 | `9d5956533843dc9aedb17923fa8583dc49494487043b1fe73d23f33ea510e9e3` |
| [`build213_22c9.collapsed.log`](build213_22c9.collapsed.log) | 380,185 | 3,589 | `b73fd1d909724567bbc8d1a961056307a7a5bfa6a3788d07c2fa3d6c9a911aa1` |

The collapsed log replaces each run of consecutive `warning C4819` lines with one `C4819 x N` line and keeps every
other line byte for byte (CRLF), ending with `C4819 total: 13764` (17,176 - 13,764 + 176 + 1 = 3,589 lines). For
publication it was redacted byte-exactly: every occurrence of the private Windows home path became
`<WINDOWS_HOME>` and every Codync bot UUID became `<BOT_ID>`; command order, exit codes, counts and line
endings are unchanged.
No line contains `error` or `fail` (case-insensitive). The only warning is MSVC C4819, a character outside the laptop's
GBK code page in nvcc's host pass: 13,764 lines in 176 runs, one after each nvcc fatbin step (170 runs of 78 lines,
4 of 82, 2 of 88). `Build Summary` is on collapsed line 1073 and `EXIT=0` on line 3587.
