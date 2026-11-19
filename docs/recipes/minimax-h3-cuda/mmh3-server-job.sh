#!/bin/sh
# mmh3 at 9c0652f as a warm server on TOP's GB10, beside TensorFold's warm picture of bench-script.sh:
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/mmh3-server-job.sh"
# The job has no network: mmh3-stage.sh put mmh3's source, its vendored crates and Rust 1.98 in the pilot folder
# as mmh3-9c0652f, beside models/MiniMax-H3-Image-mmh3 (Comfy-Org/MiniMax-H3-Image rev 0bb149f5: the INT8 ConvRot
# DiT and text encoder and the FP16 image VAE). A plain server renders the run-script.sh prompt at 2048x2048,
# seed 42, mmh3's defaults (the checkpoint's six distilled sigmas, dense BF16 attention, INT8 linears): one cold
# picture, three warm repeats of the same prompt (its states kept), then one 1024x1024 request, the size mmh3's
# docs/server.md quotes its 9.5 s for. A second server under the CUPTI preload renders one cold and two warm
# pictures whose kernel trace mmh3_segments.py splits at the >400 ms idle gaps; the last segments are the warm
# pictures. In /work/out/mmh3-server: each request's verdict and seconds, the servers' logs with mmh3's own
# stage times, the PNGs, cupti-trace.csv and cupti-warm-segment.txt. The server build adds the vendored
# axum/tokio `server` feature; requests go through python3's urllib, curl is not needed. The CUPTI preload
# replaces nsys, which is absent from the job image.
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
# The staged gcc (from toolchain-stage.sh, unpacked by cuda-build.sh or here) compiles the CUPTI
# preload for kernel tracing. nsys is absent from the job image.
gccdir=/work/arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
if [ ! -f "$gccdir/bin/aarch64-none-linux-gnu-gcc" ]; then
    tar -C /work -xJf /assets/drama3-pilot/cuda-build-toolchain/arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu.tar.xz
fi
gcc=$gccdir/bin/aarch64-none-linux-gnu-gcc
if "$gcc" -shared -fPIC -o /work/cupti-trace.so docs/recipes/minimax-h3-cuda/cupti-trace.c \
    -I/usr/local/cuda/targets/sbsa-linux/include -L/usr/local/cuda/targets/sbsa-linux/lib -lcupti 2>/dev/null; then
    echo "cupti-trace: compiled"
else
    echo "cupti-trace: COMPILE FAILED"
fi
mkdir -p "$out/jobs-timing" "$out/jobs-cupti"
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

# Phase 2: the same warm pictures under the CUPTI preload.
if [ ! -f /work/cupti-trace.so ]; then
    echo "cupti-trace not built: no kernel trace taken" >"$out/cupti-warm-segment.txt"
    exit 0
fi
CUPTI_TRACE_OUT="$out/cupti-trace.csv" LD_PRELOAD=/work/cupti-trace.so \
    "$mmh3" server --models "$models" --jobs "$out/jobs-cupti" >"$out/server-cupti.log" 2>&1 &
cupti_server=$!
trap 'kill $cupti_server 2>/dev/null || :' EXIT
python3 /work/mmh3_request.py ready
for run in 1 2 3; do
    printf %s "$prompt_text" | python3 /work/mmh3_request.py image 2048 2048 42 "$out/cupti-2048-$run.png" \
        >>"$out/requests-cupti.log" 2>&1 || { cat "$out/requests-cupti.log" >&2; exit 1; }
    echo "cupti 2048 run $run done"
done
kill "$cupti_server" 2>/dev/null || :
wait "$cupti_server" 2>/dev/null || :
cupti_server=
trap - EXIT

cat >/work/mmh3_segments.py <<'PY'
# Split the CUPTI kernel trace at >400 ms idle gaps; the last thick segments are the warm pictures.
import csv, sys
rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["start_ns"]), int(r["end_ns"]), r["name"],
                     "%sx%sx%s" % (r["gridX"], r["gridY"], r["gridZ"])))
rows.sort()
segments, current = [], []
for start, end, name, grid in rows:
    if current and start - current[-1][1] > 400_000_000:
        segments.append(current)
        current = []
    current.append((start, end, name, grid))
if current:
    segments.append(current)
with open(sys.argv[2], "w") as f:
    f.write("kernel trace: %d launches in %d segments at >400 ms idle gaps\n\n" % (len(rows), len(segments)))
    warm = []
    for i, seg in enumerate(segments):
        span = (seg[-1][1] - seg[0][0]) / 1e9
        f.write("segment %d: %d launches, %.2f s\n" % (i, len(seg), span))
    # The warm segment is the last one thick enough to be a picture's six steps.
    for seg in reversed(segments):
        if len(seg) > 50 and (seg[-1][1] - seg[0][0]) / 1e9 > 5.0:
            warm = seg
            break
    if not warm:
        f.write("\nno segment thick enough for a warm picture\n")
    else:
        f.write("\nwarm segment: %d launches, %.2f s\n\n" % (len(warm), (warm[-1][1] - warm[0][0]) / 1e9))
        f.write("%-10s %8s %10s %10s %s\n" % ("grid", "calls", "total ms", "mean us", "kernel"))
        totals = {}
        for start, end, name, grid in warm:
            key = (name, grid)
            calls, total = totals.get(key, (0, 0))
            totals[key] = (calls + 1, total + end - start)
        for (name, grid), (calls, total) in sorted(totals.items(), key=lambda kv: -kv[1][1])[:40]:
            f.write("%-10s %8d %10.1f %10.1f %s\n" % (grid, calls, total / 1e6, total / 1e3 / calls, name))
PY
if [ -f "$out/cupti-trace.csv" ]; then
    python3 /work/mmh3_segments.py "$out/cupti-trace.csv" "$out/cupti-warm-segment.txt" \
        || echo "segment split failed" >"$out/cupti-warm-segment.txt"
else
    echo "no cupti-trace.csv" >"$out/cupti-warm-segment.txt"
fi
echo "mmh3-server-job done"
