# Limits refresh receipt, 23 Nov

Job `top-57fa0d391dec` ran [`limits-job.sh`](limits-job.sh) on TOP at `4db907f60104076f8e12f4c4840c7f1ecfa9b8b4`
(main `2945c7d29` plus that script). nvcc 13.0 built the images. The audit reads each launch's registers, static
shared bytes and launch bounds from the cubin ELF's `.nv.info` and `.nv.shared` sections (`cubin.zig`); the build
does not pass `-Xptxas -v`. The job's output is [`limits-top-57fa0d391dec.txt`](limits-top-57fa0d391dec.txt). It
found 41 stale launches across 12 families, and its 10 planted control checks went RED. Each `RECORD` line was copied
unedited into `zig/src/core/contract/measurements/main_130.zig`.

Why each family moved:

| Family | Launches | Reason |
| --- | --- | --- |
| drama3 | 2 | merged change: #32 edited `drama3.cu` after `d3_cfg_mix` was recorded; #115 recorded `d3_router_compact` |
| exl3_experts | 4 | merged change: #52 left `tf_exl3_prompt_gu` and `tf_exl3_prompt_down` pending and #115 added their two k6 launches unmeasured |
| flashnext_hcn | 3 | unknown: recorded in #108, source unchanged since |
| flashnext_mtp | 1 | merged change: #57 folded the non-finite checks into `fn_row_topk` after its record |
| glm_glue | 1 | merged change: #87 and #122 edited `tf_glm_dsel_merge` in `glm_index_decode.cuh` after #57 recorded it |
| muse_dflash2 | 2 | merged change: #87 recorded only nvcc 13.3's sm_121 values |
| muse_glue | 4 | merged change: #87 recorded only nvcc 13.3's sm_121 values |
| muse_linear | 17 | merged change: #87 recorded only nvcc 13.3's sm_121 values |
| nemo_audio | 2 | toolchain: #74 recorded nvcc 13.2's values, and nvcc 13.0 builds these two launches differently |
| qmm_prefill | 1 | merged change: #11 moved shared device primitives into `cuda/common` after the record |
| qmmf | 1 | unknown: recorded in #141, source unchanged since |
| qwen35 | 3 | merged change: #11 rewrote the GDN slot scan in `qwen35_slots.cuh` after the records; #53 edited it again |

The four EXL3 prompt launches were reported as pending at every release (`exl3_prompt_pending.zig`), so their 13.0
receipt only applies because `usageFor` now checks 13.0's receipts first. Other releases keep them pending.
