# CUDA build and kernel-contract-limits in one TOP job at `cf2da2d92`

TOP's Spark (GB10, CUDA 13.0), GPU job `top-7ccced985b0a` at `cf2da2d92b453cdcb2a54c5f1387e59f3cfe4193`, started 16 Nov
2026 12:14 and exited 1 after about 19 minutes:

```sh
ssh top-menu "job start gpu top plotarmordev/TensorFold cf2da2d92b453cdcb2a54c5f1387e59f3cfe4193 docs/recipes/minimax-h3-cuda/cuda-build.sh"
```

The commit adding this receipt changes only documentation, so the run covers every script at the branch head.

[`cuda-build.sh`](../../recipes/minimax-h3-cuda/cuda-build.sh) checked and unpacked the staged Zig 0.17.0 and Arm GNU
14.3 archives, built every kernel image and `tf-cuda-test` with no network, and ran `kernel-contract-limits` over the
sm_80, sm_86, sm_89, sm_120 and sm_121 images. The job output held `tensorfold-fatbins-SHA` (115 sm_121 images, each
matching `SHA256SUMS`) and `tensorfold-cuda-SHA` (`tf-cuda-test` and 152 five-SM images). The job's last lines:

```text
FAIL kernel-contract-limits: 28 problems
FAIL kernel-contract-limits: TestFailed
kernel-contract-limits exit 1
```

Raw: [`limits.log`](limits.log). 5,420 PASS lines; both planted controls red on all five SMs; no `FAIL limits` or
`MISSED` line.
The 28 problems are stale nvcc 13.0 records, each launch still within its limits: `flashnext_mtp fn_row_topk`,
`exl3_experts tf_exl3_prompt_gu` and `tf_exl3_prompt_down`, `drama3 d3_router_compact` and `d3_cfg_mix` on all five
SMs, and `qwen35 tf_q35_gdn_slots_1`, `_2` and `_4` on sm_80. Their `RECORD` lines hold the measured usage.
