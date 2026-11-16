# Spark builds run on TOP, never on the DeepSeek Sparks

- `docs/recipes/minimax-h3-cuda/cuda-build.sh` builds a commit's CUDA kernel images and `tf-cuda-test` in one `job start gpu top` job, with Arm's GNU toolchain 14.3 as nvcc's host compiler, and runs `kernel-contract-limits` over the five release SMs' images; the sm_121 images (with `SHA256SUMS`, for `-Dfatbins`) and the runner land in the job output for later TOP jobs. `toolchain-stage.sh` stages the Zig and gcc archives in TOP's pilot folder through the Desk's `put top`.
- `docs/receipts/spark-top-build-cf2da2d92/` holds that job's run on TOP (`top-7ccced985b0a`): the build passed and `kernel-contract-limits` exited 1 on 28 stale nvcc 13.0 records, with its full `limits.log`.
- `docs/receipts/drama3-pruned-main/prebuild-fatbins.sh`, which built on BOTTOM through `bottom shell`, is removed; the Drama 3 run script and `mmh3-stage.sh` now point at TOP, as BOTTOM and MIDDLE serve DeepSeek.
- BOTTOM cleanup removed card 73's `~/opt/zig-aarch64-linux-0.17.0`, `~/scratch/tf73` and tmux sessions; 1.4 TB was free afterward, with no tmux sessions left. The pre-cleanup free-space figure was not retained.
