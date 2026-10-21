#!/usr/bin/env python3
"""Per-request GPU time split from an nsys sqlite export (`nsys export --type sqlite` of a profiled server).

Segments the kernel timeline into requests by idle gaps > GAP_S, then classifies each kernel:
  tower   : RADIO ViT + projector kernels (tf_vit_*, vision_kernel, nvjet/cutlass GEMMs, tf_rms_rows_bf16 projector norm)
  prompt  : Nemotron prompt-pass kernels (tf_ck*_prompt, tf_ck8_128*, scan, qmmf 64, pattn, router_tiles, ...)
  decode  : few/16-row decode kernels (eager first rounds; captured graphs show as GRAPH_TRACE, not kernels)
  other   : everything else
and prints per-request: wall span, busy time per class, idle inside the span, image count.
"""
import sqlite3, sys, collections

GAP_S = 0.30  # idle longer than this separates requests (a request's own idle is ms-scale)

TOWER = ("tf_vit_", "tf_prefill_attention::vision_kernel", "nvjet_", "cutlass::Kernel2", "tf_rms_rows_bf16")
DECODE = ("_few", "tf_ck8_16", "qmmf_kernel<(int)4, (int)16", "tf_super_topk_few", "tf_nemo_router_rows",
          "tf_super_combine", "tf_super_moe_out", "tf_nemotron_mamba_a", "tf_argmax", "plan_kernel", "tf_nemo_kv_write",
          "tf_nemo_concat_norms", "tf_pack_bf16_lane", "tf_nemo_conv_commit")


def cls(name):
    for p in TOWER:
        if p in name:
            return "tower"
    for p in DECODE:
        if p in name:
            # tf_super_combine_sums / moe_out_sums are prompt kernels; exact names below
            if name in ("tf_super_combine_sums", "tf_super_moe_out_sums"):
                return "prompt"
            return "decode"
    return "prompt"


def main(path):
    db = sqlite3.connect(path)
    rows = db.execute(
        "select k.start, k.end, s.value, k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.demangledName order by k.start").fetchall()
    graphs = db.execute("select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start").fetchall()
    t0 = rows[0][0]
    # segment
    reqs, cur, last_end = [], [], None
    for st, en, nm, sid in rows:
        if last_end is not None and st - last_end > GAP_S * 1e9:
            reqs.append(cur)
            cur = []
        cur.append((st, en, nm, sid))
        last_end = max(last_end or en, en)
    reqs.append(cur)
    print(f"{len(rows)} kernels, {len(graphs)} graph launches, {len(reqs)} segments (gap > {GAP_S}s)")
    print(f"{'seg':>3} {'t0 s':>8} {'span s':>7} {'tower':>7} {'prompt':>7} {'decode':>7} {'busy':>7} {'idle':>7} "
          f"{'imgs':>4} {'vitattn':>8} {'nvjet':>6} {'ck4dn':>6} {'graphs':>6} {'gspan':>6}")
    for i, r in enumerate(reqs):
        s0, s1 = r[0][0], max(e for _, e, _, _ in r)
        by = collections.Counter()
        cnt = collections.Counter()
        # union busy time across streams
        ivs = sorted((a, b) for a, b, _, _ in r)
        busy, ce = 0, -1
        for a, b in ivs:
            if a > ce:
                busy += b - a; ce = b
            elif b > ce:
                busy += b - ce; ce = b
        for a, b, nm, _ in r:
            by[cls(nm)] += b - a
            if "vision_kernel" in nm: cnt["vitattn"] += 1
            if "nvjet" in nm: cnt["nvjet"] += 1
            if nm == "tf_ck4_down_sum_prompt": cnt["ck4dn"] += 1
            if nm == "tf_vit_position_add": cnt["imgs"] += 1
        g = [(a, b) for a, b in graphs if a >= s0 and a <= s1 + 2e9]
        gspan = (max(b for _, b in g) - min(a for a, _ in g)) / 1e9 if g else 0
        print(f"{i:>3} {(s0-t0)/1e9:>8.2f} {(s1-s0)/1e9:>7.3f} {by['tower']/1e9:>7.3f} {by['prompt']/1e9:>7.3f} "
              f"{by['decode']/1e9:>7.3f} {busy/1e9:>7.3f} {(s1-s0-busy)/1e9:>7.3f} {cnt['imgs']:>4} {cnt['vitattn']:>8} "
              f"{cnt['nvjet']:>6} {cnt['ck4dn']:>6} {len(g):>6} {gspan:>6.2f}")
    if len(sys.argv) > 2 and sys.argv[2] == "detail":
        seg = int(sys.argv[3])
        r = reqs[seg]
        s0 = r[0][0]
        prev_end = s0
        for a, b, nm, sid in r:
            gap = (a - prev_end) / 1e9
            print(f"{(a-s0)/1e9:9.4f} +{(b-a)/1e6:8.3f}ms gap {gap*1e3:7.2f}ms s{sid:<3} {nm[:90]}")
            prev_end = max(prev_end, b)


if __name__ == "__main__":
    main(sys.argv[1])
