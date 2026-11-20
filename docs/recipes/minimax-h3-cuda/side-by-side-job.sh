#!/bin/sh
# TensorFold's and mmh3's warm servers render the same five 2048² prompts back to back on TOP's
# GB10 in one job — the card's side-by-side:
#   swarm gpu-run plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/side-by-side-job.sh
# The prompt list is the pilot's fox (episode 1's gate prompt), the three files in ./prompts/ that
# neither server has seen, then the fox once more. Runs 1-4 of each engine are cold-then-unseen;
# run 5 is the repeated-prompt number, reported separately and labelled, never part of the
# headline. Seed 42 and the checkpoint's six distilled steps throughout; mmh3's defaults are those
# six sigmas, dense BF16 attention and its INT8 ConvRot linears, ours the default build (qkvc and
# fc1 int8, the tensor-core decoder).
# Phase 1 is bench-script.sh itself under H3_BENCH_PROMPTS: its cold render warms the three
# resident servers, runs 2-4 are the unseen warm runs, run 5 the labelled repeat, and the
# selftests plus the same-latent `reference` compare gate it as ever. Phase 2 is
# mmh3-server-job.sh's timing half (no CUPTI trace): `mmh3 server` answers the same five prompts
# in the same order. Our numbers land in /work/out/h3-side-by-side (its SUMMARY.md), mmh3's in
# /work/out/mmh3-side-by-side: each request's client wall, the server's own generation report,
# the PNGs and SUMMARY.md's per-run table.
set -eu
here=docs/recipes/minimax-h3-cuda
pilot=/assets/drama3-pilot
stage=$pilot/mmh3-9c0652f
models=$pilot/models/MiniMax-H3-Image-mmh3
pilot_prompt=$pilot/h3-image-pilot/prompt.txt
list="$pilot_prompt $here/prompts/harbour.txt $here/prompts/meadow.txt $here/prompts/steam.txt $pilot_prompt"
out=/work/out/mmh3-side-by-side
for path in "$pilot_prompt" "$here/prompts/harbour.txt" "$here/prompts/meadow.txt" "$here/prompts/steam.txt" \
    "$models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors" \
    "$models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    "$models/vae/minimax_h3_image_vae_fp16.safetensors" "$stage/src/vendor"; do
    [ -e "$path" ] || { echo "missing $path" >&2; exit 1; }
done

# Phase 1: our bench, run n on the nth prompt of the list.
H3_BENCH_OUT=/work/out/h3-side-by-side H3_BENCH_PROMPTS="$list" sh "$here/bench-script.sh"
printf '%s\n' $list > /work/out/h3-side-by-side/run-prompts.txt

# Phase 2: mmh3's server, the same list in the same order.
mkdir -p "$out/jobs"
export PATH=$stage/rustup/toolchains/1.98.0-aarch64-unknown-linux-gnu/bin:$PATH CARGO_HOME=/work/cargo \
    CARGO_TARGET_DIR=/work/mmh3-target CUDA_HOME=/usr/local/cuda MMH3_CUDA_ARCH=sm_120f \
    LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
(cd "$stage/src" && cargo build --release --offline --locked --features cuda,server --bin mmh3) \
    >"$out/build.log" 2>&1 || { tail -40 "$out/build.log" >&2; exit 1; }
mmh3=/work/mmh3-target/release/mmh3

cat >/work/mmh3_request.py <<'PY'
# request.py ready | image <width> <height> <seed> <out.png>; the prompt on stdin.
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

"$mmh3" server --models "$models" --jobs "$out/jobs" >"$out/server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || :' EXIT
python3 /work/mmh3_request.py ready
run=0
for file in $list; do
    run=$((run + 1))
    date +%s.%N >"$out/2048-$run.time"
    python3 /work/mmh3_request.py image 2048 2048 42 "$out/2048-$run.png" <"$file" \
        >>"$out/requests.log" 2>&1 || { cat "$out/requests.log" >&2; exit 1; }
    date +%s.%N >>"$out/2048-$run.time"
    echo "mmh3 run $run ($file) done"
done
kill "$server" 2>/dev/null || :
wait "$server" 2>/dev/null || :
server=
trap - EXIT

cat >/work/mmh3_summary.py <<'PY'
# mmh3's side of the window: a run's client wall beside every seconds field its generation carries.
import json
from pathlib import Path
out = Path("/work/out/mmh3-side-by-side")
lines = ["# mmh3 server, five 2048^2 requests, seed 42, six steps", "",
         "Runs 1-4: cold, then three prompts the server has not seen. Run 5 repeats run 1's prompt —",
         "the kept-states number, reported separately, not part of the unseen-prompt headline.", "",
         "| run | prompt | client wall s | generation report |", "|---:|---|---:|---|"]
prompts = (out / "run-prompts.txt").read_text().split()
reports = (out / "requests.log").read_text().splitlines()
for n, line in enumerate(reports, 1):
    t = (out / f"2048-{n}.time").read_text().split()
    wall = f"{float(t[1]) - float(t[0]):.2f}" if len(t) == 2 else "—"
    st = json.loads(line.split(":", 1)[1])
    fields = ", ".join(f"{k} {st[k]}" for k in sorted(st) if any(w in k for w in ("second", "duration", "elapsed", "time")))
    lines.append(f"| {n} | {Path(prompts[n - 1]).name} | {wall} | {fields or 'state ' + st['state']} |")
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
PY
printf '%s\n' $list > "$out/run-prompts.txt"
python3 /work/mmh3_summary.py
echo "side-by-side job done"
