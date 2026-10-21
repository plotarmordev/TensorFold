# Receipts: radio CUDA: reuse one pinned pixel stage for every image

Branch tip `25b565cc8cb55e5d8764f510054e2f8ce1e04082` on main `d1018b318fb51981c4a18f12e40269a4d3ee9713`.

Builds (from detached checkouts; `env/PROVENANCE-*.txt` give the commits, flags, toolchain, diff stat and binary
sha256; `env/commits.txt` gives every measured and submitted commit's parent, tree and `git patch-id --stable`):

| label | commits | `tensorfold-native` sha256 |
|---|---|---|
| `v0` (base) | main + #1057 (`732b0e31`) | `5e70f78f8029df2c257c997db0e5cc54c1ace12f807982d23efe980d9fdcef69` |
| `vp` (this branch) | main + #1057 (`2bd8f126`) + this change (`5a4d65a1`) | `7df4fbe7672ca24889f289961af4c62f129a9875006ddc2515fba94785d56ca8` |
| `vgp` ("both") | main + #1057 (`baa109d8`) + the RADIO fused-glue change (`b7e05211`) + this change (`f6d8ba5a`) | `34fba74428d133856ec03b78dab14b5d29962eee8826d2da54fa86ed35a29cda` |

`main-decode.*` ran main itself (`7189a4cd54646d2f0f6fadd9b650a3be9211eb1565399aef5649d366f592ab0b`) for the decode
hash comparison. Flags: `zig build fatbins` / `zig build install` with `-Dtarget=aarch64-linux-gnu.2.28
-Dcpu=baseline -Doptimize=fast -Dstrip=true -Dsm=121 -Dversion=1.1.1 -Dnvcc=<nvcc>`, zig 0.17.0, nvcc 13.0.88.

One DGX Spark (GB10, sm_121), driver 580.173.02, CUDA 13.0, checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`. Every hold is a fresh
server in a fresh `HOME` with `TF_HEAT_HIGH=85 TF_HEAT_LOW=75`, started at a zone of 60 °C or less and a load under
1.5. Image holds serve `--vision --vision-image-tokens 13312 --vision-max-images 4 --context 18432 --parallel 1
--prompt-cache-gib 0`; the client is `scripts/bench.py image-ttft` with `scripts/prompts.py`, both as run. Per hold:
`.hold.log` (command, binary sha256, client output), `.client.log`, `.server.log` (one `done` line per request, with
`ttft=` and `sha=`; `info: heat_wait_s` lines are the server's heat pauses), `.guard.log` (GPU and zone temperature,
load and free memory every 5 s), `.exit` (client exit status), `.sha256` (the served binary). Paths private to the
machine are placeholders (`<home>`, `<out>`, `<checkpoint>`, `<loopback>`, `<node>`, `<scripts>`, `<tools>`,
`<checkout>`, `<cache>`, `<zig>`).

| Path | What |
|---|---|
| `gpu/trace-{v0,vp}.*` | Each server under `nsys profile --trace=cuda --sample=none --cpuctxsw=none`, v0 then vp. |
| `gpu/timeline-{v0,vp}.txt` | `scripts/timeline.py` on each trace's `nsys export --type sqlite`: one row per request (gap > 0.3 s), tower/prompt/decode busy seconds and idle inside the span. Segments with `imgs 1` / `imgs 4` are the tower of a 1-image / 4-image request (a 1-image request's tower and prompt share one segment). |
| `gpu/host-api.txt` | `scripts/host_api.py` on both exports: kernels, copies by kind with bytes, and `cuMemHostAlloc` / `cuMemFreeHost` before and from 0.2 s before the first ViT kernel. The `.nsys-rep` and `.sqlite` files are not included (17 and 44 MB each). |
| `gpu/image-{0-v0,1-vgp,2-v0,3-vgp}.*` | Unprofiled image holds in that order (this commit together with the fused-glue commit). |
| `gpu/{vgp,main}-decode.*` | `bench.py decode`: 11 prompts drafted (count, prose, code, image with one 320 x 208 picture, multilingual, six languages), each against `"draft": false`, plus a count row with `"draft": false`; `--context 131072 --parallel auto --vision --thinking`. |
| `env/commits.txt` | Parent, tree and `git patch-id --stable` of every measured build commit and of the submitted commits, with the script that printed them. |
| `host/source-history.txt` | `git merge-base --is-ancestor` for the two commits that moved `uploadAsync` onto temporary stages, the main source lines of that path, and the stage size for the measured image. |
| `host/fail-before.txt` | The new test with the tower's original `scratch.uploadAsync` line: `expected 1, found 2` (transcript). |
| `host/review-focused.txt` | At this tip: `zig build test-bolt-load-host -Dbolt-load-filter='RADIO upload' test-cuda-copy-source -j1 --summary all`. |
| `host/review-mutation.txt` | The same command on a copy of this tip whose `features` writes the next image into the stage before `take`: the test aborts at the in-flight byte comparison. |
| `host/review-host.txt` | At this tip: `zig build test -j1 --summary all`. |
| `host/review-fmt.exit` | `zig fmt --check` on the three changed Zig files (no output). |
| `host/review-lean-{base,head,new}.txt` | `tools/zig/lean_check.py` on main and this tip, `tools/zig/lean_new.py` on the two (exit 1 of the first two is the checker's status with problems present). |
| `host/tip-gates.txt` | At this tip: `zig build test-kernel-contract test-golden fuzz-cortheon -j1 --summary all`. |
| `scripts/` | `bench.py`, `prompts.py` (image client), `timeline.py` (docstring path generalised), `host_api.py`. |
| `SHA256SUMS` | sha256 of every other file here. |
