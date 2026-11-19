#!/bin/sh
# mmh3 at 9c0652f as a warm server on TOP's GB10, beside TensorFold's warm picture of bench-script.sh:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/mmh3-server-job.sh"
# The job has no network: mmh3-stage.sh put mmh3's source, its vendored crates and Rust 1.98 in the pilot folder
# as mmh3-9c0652f, beside models/MiniMax-H3-Image-mmh3 (Comfy-Org/MiniMax-H3-Image rev 0bb149f5: the INT8 ConvRot
# DiT and text encoder and the FP16 image VAE). A plain server renders the run-script.sh prompt at 2048x2048,
# seed 42, mmh3's defaults (the checkpoint's six distilled sigmas, dense BF16 attention, INT8 linears): one cold
# picture, three warm repeats of the same prompt (its states kept), then one 1024x1024 request, the size mmh3's
# docs/server.md quotes its 9.5 s for. A second server under nsys renders one cold and two warm pictures whose
# kernel trace mmh3_segments.py splits at the >400 ms idle gaps; the last segments are the warm pictures. In
# /work/out/mmh3-server: each request's verdict and seconds, the servers' logs with mmh3's own stage times, the
# PNGs, nsys-stats.txt, nsys-warm-segment.txt and mmh3-server.sqlite (the multi-GB .nsys-rep is dropped). The
# server build adds the vendored axum/tokio `server` feature; requests go through python3's urllib, curl is not
# needed. ncu, if the image has it, profiles the top kernel of the warm segment on a cold `mmh3 image`.
set -eu
pilot=/assets/drama3-pilot
stage=$pilot/mmh3-9c0652f
models=$pilot/models/MiniMax-H3-Image-mmh3
prompt=$pilot/h3-image-pilot/prompt.txt
out=/work/out/mmh3-server
for path in "$models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors" \
    "$models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    "$models/vae/minimax_h3_image_vae_fp16.safetensors" "$stage/src/vendor" "$prompt"; do
    [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
for tool in nsys ncu python3; do
    if command -v "$tool" >/dev/null 2>&1; then echo "$tool: $(command -v "$tool")"; else echo "$tool: MISSING"; fi
done
mkdir -p "$out/jobs-timing" "$out/jobs-nsys"
export PATH=$stage/rustup/toolchains/1.98.0-aarch64-unknown-linux-gnu/bin:$PATH CARGO_HOME=/work/cargo \
    CARGO_TARGET_DIR=/work/mmh3-target CUDA_HOME=/usr/local/cuda MMH3_CUDA_ARCH=sm_120f \
    LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
(cd "$stage/src" && cargo build --release --offline --locked --features cuda,server --bin mmh3) \
    >"$out/build.log" 2>&1 || { tail -40 "$out/build.log" >&2; exit 1; }
mmh3=/work/mmh3-target/release/mmh3

cat >/work/mmh3_request.py <<'PY'
# request.py ready | image <width> <height> <seed> <out.png>
import json, sys, time, urllib.request
BASE = "http://127.0.0.1:8833"

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.read()

if sys.argv[1] == "ready":
    for _ in range(600):
        try:
            get("/v1/status")
            sys.exit(0)
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server did not answer /v1/status")

_, _, width, height, seed, out_png = sys.argv
prompt = sys.stdin.read()
boundary = "mmh3jobboundary"
body = b""
for name, value in [("prompt", prompt), ("width", width), ("height", height), ("seed", seed)]:
    body += ('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n'
             % (boundary, name, value)).encode()
body += ("--%s--\r\n" % boundary).encode()
req = urllib.request.Request(BASE + "/v1/images", data=body,
    headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
gen = json.loads(urllib.request.urlopen(req).read())
gid = gen["id"]
started = time.time()
while True:
    time.sleep(0.5)
    st = json.loads(get("/v1/generations/" + gid))
    if st["state"] in ("done", "failed"):
        break
    if time.time() - started > 900:
        raise SystemExit("generation %s still running after 900 s" % gid)
print("generation %s: %s" % (gid, json.dumps(st, sort_keys=True)))
if st["state"] != "done":
    sys.exit(1)
open(out_png, "wb").write(get("/v1/generations/%s/image" % gid))
PY

# Phase 1: clean timing, no profiler.
"$mmh3" server --models "$models" --jobs "$out/jobs-timing" >"$out/server-timing.log" 2>&1 &
timing_server=$!
trap 'kill $timing_server 2>/dev/null || :' EXIT
python3 /work/mmh3_request.py ready
prompt_text=$(cat "$prompt") # $(...) drops the file's last line ending, as both engines' prompt stages do
for run in 1 2 3 4; do
    printf %s "$prompt_text" | python3 /work/mmh3_request.py image 2048 2048 42 "$out/2048-$run.png" \
        >>"$out/requests.log" 2>&1 || { cat "$out/requests.log" >&2; exit 1; }
    echo "2048 run $run done"
done
printf %s "$prompt_text" | python3 /work/mmh3_request.py image 1024 1024 42 "$out/1024-1.png" \
    >>"$out/requests.log" 2>&1 || { cat "$out/requests.log" >&2; exit 1; }
echo "1024 run done"
kill "$timing_server" 2>/dev/null || :
wait "$timing_server" 2>/dev/null || :
timing_server=
trap - EXIT

# Phase 2: the same warm pictures under nsys, when the image has it.
if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys MISSING: no kernel trace taken" >"$out/nsys-stats.txt"
    exit 0
fi
nsys profile --trace=cuda,nvtx,cublas --force-overwrite=true -o "$out/mmh3-server" \
    "$mmh3" server --models "$models" --jobs "$out/jobs-nsys" >"$out/server-nsys.log" 2>&1 &
nsys_server=$!
trap 'kill $nsys_server 2>/dev/null || :' EXIT
python3 /work/mmh3_request.py ready
for run in 1 2 3; do
    date +%s.%N >>"$out/nsys-windows.txt"
    printf %s "$prompt_text" | python3 /work/mmh3_request.py image 2048 2048 42 "$out/nsys-2048-$run.png" \
        >>"$out/requests-nsys.log" 2>&1 || { cat "$out/requests-nsys.log" >&2; exit 1; }
    date +%s.%N >>"$out/nsys-windows.txt"
    echo "nsys 2048 run $run done"
done
# the server's own PID ends the trace: nsys finalizes the report when its child exits; SIGKILL on
# nsys would drop it. pgrep is not a given, so /proc's comms name the mmh3 process.
server_pid=
for comm in /proc/[0-9]*/comm; do
    if [ "$(cat "$comm" 2>/dev/null || :)" = mmh3 ]; then
        server_pid=${comm%/comm}
        server_pid=${server_pid#/proc/}
    fi
done
[ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || :
for _ in $(seq 30); do
    kill -0 "$nsys_server" 2>/dev/null || break
    sleep 1
done
kill -9 "$nsys_server" 2>/dev/null || : # still alive after 30 s: the report may be lost, checked below
wait "$nsys_server" 2>/dev/null || :
nsys_server=
trap - EXIT
[ -f "$out/mmh3-server.nsys-rep" ] || { echo "nsys wrote no report" >"$out/nsys-stats.txt"; exit 0; }
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
    "$out/mmh3-server.nsys-rep" >"$out/nsys-stats.txt" 2>&1 || :
nsys export --type sqlite --force-overwrite=true -o "$out/mmh3-server.sqlite" \
    "$out/mmh3-server.nsys-rep" >/dev/null 2>&1 || :

cat >/work/mmh3_segments.py <<'PY'
# Split the kernel trace at >400 ms idle gaps; the last thick segments are the two warm pictures.
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = []
for (table,) in db.execute("select name from sqlite_master where type='table'"):
    if "KERNEL" in table.upper():
        cols = [c[1] for c in db.execute("pragma table_info(%s)" % table)]
        if "start" in cols and "end" in cols and ("demangledName" in cols or "shortName" in cols):
            col = "demangledName" if "demangledName" in cols else "shortName"
            rows = db.execute("select start, end, %s, gridX, gridY, gridZ from %s" % (col, table)).fetchall()
            break
rows.sort()
segments, current = [], []
for start, end, name, gx, gy, gz in rows:
    if current and start - current[-1][1] > 400_000_000:
        segments.append(current)
        current = []
    current.append((start, end, name, gx, gy, gz))
if current:
    segments.append(current)
out = sys.argv[2]
with open(out, "w") as f:
    f.write("kernel trace: %d launches in %d segments at >400 ms idle gaps\n\n" % (len(rows), len(segments)))
    warm = []
    for i, seg in enumerate(segments):
        span = (seg[-1][1] - seg[0][0]) / 1e9
        f.write("segment %d: %d launches, %.2f s\n" % (i, len(seg), span))
        if span > 3.0:
            warm = seg  # the last thick segment: the second warm picture
    f.write("\nwarm segment: %d launches, %.2f s\n\n" % (len(warm), (warm[-1][1] - warm[0][0]) / 1e9))
    f.write("%-10s %8s %10s %10s %s\n" % ("grid", "calls", "total ms", "mean us", "kernel"))
    totals = {}
    for start, end, name, gx, gy, gz in warm:
        key = (names.get(name, str(name)), "%dx%dx%d" % (gx, gy, gz))
        calls, total = totals.get(key, (0, 0))
        totals[key] = (calls + 1, total + end - start)
    for (name, grid), (calls, total) in sorted(totals.items(), key=lambda kv: -kv[1][1])[:40]:
        f.write("%-10s %8d %10.1f %10.1f %s\n" % (grid, calls, total / 1e6, total / 1e3 / calls, name))
    with open(sys.argv[3], "w") as kf:
        for (name, grid), (calls, total) in sorted(totals.items(), key=lambda kv: -kv[1][1])[:3]:
            kf.write(name + "\n")
PY
if [ -f "$out/mmh3-server.sqlite" ]; then
    python3 /work/mmh3_segments.py "$out/mmh3-server.sqlite" "$out/nsys-warm-segment.txt" "$out/top-kernels.txt" \
        || echo "segment split failed" >"$out/nsys-warm-segment.txt"
else
    echo "sqlite export failed" >"$out/nsys-warm-segment.txt"
fi
rm -f "$out/mmh3-server.nsys-rep"

# Phase 3: ncu on the warm segment's top kernels, if the image has it (replays ~6 launches each).
if command -v ncu >/dev/null 2>&1 && [ -f "$out/top-kernels.txt" ]; then
    n=0
    while IFS= read -r kernel && [ "$n" -lt 3 ]; do
        n=$((n + 1))
        ncu --kernel-name "regex:$kernel" --launch-count 6 \
            --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
            "$mmh3" image --models "$models" --prompt "$prompt_text" --seed 42 \
            --width 2048 --height 2048 --out "$out/ncu.png" >"$out/ncu-$n.txt" 2>&1 || :
        rm -f "$out/ncu.png"
    done <"$out/top-kernels.txt"
fi
echo "mmh3-server-job done"
