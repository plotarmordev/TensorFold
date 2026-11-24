#!/bin/sh
# kernel-contract-limits over every image of the five release SMs, as a TOP job:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/kernel-contract/limits-job.sh
# nvcc builds the images with -Xptxas -v (one thread per SASS target), the build stopping at 600 s. Prints each stale
# or failing check and the RECORD line of its launch (the values its record is replaced with), the router's measured
# lines, then ptxas's registers and static shared bytes per SM for the router and the 41 launches main_130.zig records.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
symbols="tf_nemo_router_tiles d3_cfg_mix d3_router_compact dfl_topk dfl_walk fn_hcn_down2 fn_hcn_down3 fn_hcn_up3
fn_row_topk muse_argmax muse_expert_sum muse_linear2_k2_w4 muse_linear2_k2_w8 muse_linear2_k3_w8 muse_linear2_k4_w4
muse_linear2_k4_w8 muse_linear2_k5_w4 muse_linear2_k5_w8 muse_linear2_k6_w4 muse_linear2_k6_w8 muse_linear2_k8_w4
muse_linear_multi_k2 muse_linear_multi_k3 muse_linear_multi_k4 muse_linear_multi_k5 muse_linear_multi_k6
muse_linear_multi_k8 muse_rms muse_rot_multi muse_softcap tf_exl3_prompt_down tf_exl3_prompt_down_k6 tf_exl3_prompt_gu
tf_exl3_prompt_gu_k6 tf_glm_dsel_merge tf_nemo_attn64 tf_nemo_preds tf_q35_gdn_slots_1 tf_q35_gdn_slots_2
tf_q35_gdn_slots_4 tf_qmm_prefill_many _ZN7tf_qmmf11qmmf_kernelILi4ELi16ELi64ELi1ELi4ELi4ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii"
status=0
NVCC_APPEND_FLAGS="-Xptxas -v --threads 0" H3_BUILD=limits LIMITS_IMAGES= timeout 600 \
    sh "$here/../recipes/minimax-h3-cuda/cuda-build.sh" >/work/out/build.log 2>&1 || status=$?
[ "$status" = 124 ] && echo "limits job stopped at its 600 s limit"
tail -n 4 /work/out/build.log
log=/work/out/limits.log
grep -E '^(INFO|SUMMARY|MISSED|RED )' "$log"
awk '/^(PASS|FAIL) limits sm_[0-9]+ .*\(record stale\)|^FAIL limits/ { sub(/^(PASS|FAIL) limits sm_[0-9]+ /, ""); sub(/: [0-9]+ registers,.*/, ""); stale[$0] = 1; print "STALE " $0 }
    /^RECORD / { name = substr($0, 8); sub(/: \.usage = .*/, "", name); if (name in stale) print }' "$log" | sort -u
grep -E '^(PASS|FAIL) limits sm_[0-9]+ router tf_nemo_router_tiles:' "$log"
symbols=$symbols awk -F"'" 'BEGIN { n = split(ENVIRON["symbols"], s, /[ \n]+/); for (i = 1; i <= n; i++) want[s[i]] = 1 }
    /Compiling entry function/ { fn = $2; sm = $4; next }
    fn != "" && /ptxas info *: Used / { if (fn in want) {
        regs = $0; sub(/.*Used /, "", regs); sub(/ registers.*/, "", regs)
        smem = "0"; if (match($0, /[0-9]+ bytes smem/)) smem = substr($0, RSTART, RLENGTH - 11)
        use[fn] = use[fn] " " sm "=" regs "r/" smem "s" }
        fn = "" }
    END { for (f in use) print "PTXAS " f ":" use[f] }' /work/out/build.log | sort
exit "$status"
