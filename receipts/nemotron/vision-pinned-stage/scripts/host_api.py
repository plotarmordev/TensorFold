#!/usr/bin/env python3
"""Pinned host alloc/free calls and copy totals in an nsys sqlite export, split at the first ViT kernel.

  python3 host_api.py trace.sqlite
"""
import sqlite3
import sys

WINDOW_NS = 200_000_000  # the window starts this long before the first ViT kernel, to catch image 1's upload

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
first = db.execute("SELECT min(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName "
                   "WHERE s.value LIKE 'tf_vit%'").fetchone()[0]
print(f"kernels {db.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]}")
for kind, count, total in db.execute("SELECT copyKind, count(*), sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                                     "GROUP BY copyKind ORDER BY copyKind"):
    print(f"memcpy kind {kind}: {count} copies, {total} bytes")
for label, cond in (("before the images", "r.start < ?"), ("from the images on", "r.start >= ?")):
    rows = db.execute("SELECT s.value, count(*), sum(r.end - r.start) / 1e6, max(r.end - r.start) / 1e6 "
                      "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
                      f"WHERE (s.value LIKE 'cuMemHostAlloc%' OR s.value LIKE 'cuMemFreeHost%') AND {cond} "
                      "GROUP BY s.value ORDER BY s.value", (first - WINDOW_NS,)).fetchall()
    print(f"{label}: " + ("; ".join(f"{n} {c} calls, {t:.1f} ms total, max {m:.3f} ms" for n, c, t, m in rows) or "none"))
