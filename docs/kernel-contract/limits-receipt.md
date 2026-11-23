# Limits refresh receipt, 23 Nov

Job `top-57fa0d391dec` ran [`limits-job.sh`](limits-job.sh) on TOP at `4db907f60104076f8e12f4c4840c7f1ecfa9b8b4`
(main `2945c7d29` plus that script). nvcc 13.0 built the images, and its `-Xptxas -v` resources are in
[`limits-top-57fa0d391dec.log`](limits-top-57fa0d391dec.log). The job found 41 stale launches across 12 families; the
other problems in its count were the 11 planted controls going RED. Each `RECORD` line was copied unedited into
`zig/src/core/contract/measurements/main_130.zig`.

Why each family moved:

| Family | Launches | Reason |
| --- | --- | --- |
| drama3 | 2 | merged change: #32 edited `drama3.cu` after `d3_cfg_mix` was recorded; #115 recorded `d3_router_compact` |
| exl3_experts | 4 | merged change: #52 left the four prompt launches pending, never measured |
| flashnext_hcn | 3 | unknown: recorded in #108, source unchanged since |
| flashnext_mtp | 1 | merged change: #57 folded the non-finite checks into `fn_row_topk` after its record |
| glm_glue | 1 | unknown: recorded in #57, source unchanged since |
| muse_dflash2 | 2 | merged change: #87 recorded only nvcc 13.3's sm_121 values |
| muse_glue | 4 | merged change: #87 recorded only nvcc 13.3's sm_121 values |
| muse_linear | 17 | merged change: #87 recorded only nvcc 13.3's sm_121 values |
| nemo_audio | 2 | unknown: recorded in #74, source unchanged since |
| qmm_prefill | 1 | merged change: #11 moved shared device primitives into `cuda/common` after the record |
| qmmf | 1 | unknown: recorded in #141, source unchanged since |
| qwen35 | 3 | merged change: #11 specialized whole-window attention in `qwen35.cu` after the records |

The four EXL3 prompt launches were reported as pending at every release (`exl3_prompt_pending.zig`), so their 13.0
receipt only applies because `usageFor` now checks 13.0's receipts first. Other releases keep them pending.
