# EXL3 encoder threads

- `tf-exl3-encode --threads N` sets how many threads one encode uses (1 to 64; every core by default), so several encodes can share a Mac without running more threads than it has cores.
- Threads take the next 16x16 tile from a shared counter instead of a fixed band of tile rows each, so none sits idle when the rows do not divide evenly or there are fewer rows than cores; the full-Hessian errors (EXL3, plain rounding and NVFP4) also run on them, where they ran on one core.
- The encoded bytes and errors are the same at every thread count. Tests check every count from 1 to over the cap, and `--threads` parsing; the tool's tests now run in `zig build test-exl3`.
- `docs/recipes/exl3.md` says which workers and threads settings fill the Mac's 36 cores without going over.
