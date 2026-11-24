# Kernel contract: CUDA limits records match nvcc 13.0 on GB10 again

- 41 launches across 12 families had stale records. They now carry the resources nvcc 13.0 compiled on TOP
  (job `top-57fa0d391dec`), in `measurements/main_130.zig`. The reasons per family are in
  `docs/kernel-contract/limits-receipt.md`.
- `docs/kernel-contract/limits-job.sh` runs the full five-SM limits audit as a TOP job and prints each stale launch's
  `RECORD` line.
- The EXL3 prompt launches use their measured 13.0 receipt and remain pending on other releases.
