# Receipts: radio CUDA: fuse the tower's bias, activation and residual launches

Branch tip `47ef124133841184afb680c775f31e8db65e6619` on main `d1018b318fb51981c4a18f12e40269a4d3ee9713`. It has the
same tree (`2a07a5f1…`) as `c8dc633c0421e91fa822cdd1028dcfbeb23a4990`, the commit the host gates below name; only the
commit message differs.

Builds (from detached checkouts; `env/PROVENANCE-*.txt` give the commits, flags, toolchain, diff stat and binary
sha256; `env/commits.txt` gives every measured and submitted commit's parent, tree and `git patch-id --stable`):

| label | commits | sha256 |
|---|---|---|
| `v0` (base) | main + #1057 (`732b0e31`) | `tensorfold-native` `5e70f78f8029df2c257c997db0e5cc54c1ace12f807982d23efe980d9fdcef69` |
| `vg` (contract build) | main + #1057 (`e9ec039d`) + this change (`b4ae488c`) | `tf-cuda-test` `c8129813edfb5353de97ea5d01a90c842e1e2076264974845812f42179f29973` |
| `vgp` ("both") | main + #1057 (`baa109d8`) + this change (`b7e05211`) + the pinned-stage commit (`f6d8ba5a`) | `tensorfold-native` `34fba74428d133856ec03b78dab14b5d29962eee8826d2da54fa86ed35a29cda` |

`main-decode.*` ran main itself (`7189a4cd54646d2f0f6fadd9b650a3be9211eb1565399aef5649d366f592ab0b`) for the decode
hash comparison. Flags: `zig build fatbins` / `zig build install` with `-Dtarget=aarch64-linux-gnu.2.28
-Dcpu=baseline -Doptimize=fast -Dstrip=true -Dsm=121 -Dversion=1.1.1 -Dnvcc=<nvcc>`, zig 0.17.0, nvcc 13.0.88.

One DGX Spark (GB10, sm_121), driver 580.173.02, CUDA 13.0, checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`, `TF_HEAT_HIGH=85
TF_HEAT_LOW=75`, every hold started at a zone of 60 °C or less and a load under 1.5. Per hold: `.hold.log` (command,
binary sha256, output summary), `.guard.log` (GPU and zone temperature, load and free memory every 5 s), `.exit`,
`.sha256` (the binary run); server holds add `.client.log` and `.server.log` (one `done` line per request with
`ttft=` and `sha=`; `info: heat_wait_s` lines are the server's heat pauses). Paths private to the machine are
placeholders (`<home>`, `<out>`, `<checkpoint>`, `<loopback>`, `<node>`, `<scripts>`, `<tools>`, `<checkout>`,
`<cache>`, `<zig>`).

| Path | What |
|---|---|
| `gpu/glue-contract.*` | `tf-cuda-test kernel-contract` on `vg`. `.log` is the full output: the 48 `tf_vit_bias_act` gate lines are at the top, its limits line in the `limits live` section, the SUMMARY at the end. |
| `gpu/glue-memcheck.*` | The same under `compute-sanitizer --tool memcheck --error-exitcode 1`; `ERROR SUMMARY: 0 errors` at the end. |
| `gpu/limit-problems.txt` | The 1533 limit problems of `glue-contract.log` counted by CUDA image, and every line that names `tf_vit_bias_act`. |
| `gpu/sass.txt` | `cuobjdump -sass -fun` sha256 of `tf_vit_pointwise` and `tf_vit_bias` in main's and `vg`'s sm_121 `vit.fatbin` (identical), and `cuobjdump -res-usage` of `tf_vit_bias_act`, `tf_vit_pointwise`, `tf_vit_bias` in `vg`'s. |
| `gpu/image-{0-v0,1-vgp,2-v0,3-vgp}.*` | `scripts/bench.py image-ttft` (with `scripts/prompts.py`, as run), unprofiled, in that order; served `--vision --vision-image-tokens 13312 --vision-max-images 4 --context 18432 --parallel 1 --prompt-cache-gib 0`. This commit is timed only inside `vgp`. |
| `gpu/{vgp,main}-decode.*` | `bench.py decode`: 11 prompts drafted (count, prose, code, image with one 320 x 208 picture, multilingual, six languages), each against `"draft": false`, plus a count row with `"draft": false`; `--context 131072 --parallel auto --vision --thinking`. |
| `env/commits.txt` | Parent, tree and `git patch-id --stable` of every measured build commit and of the submitted commits, with the script that printed them. |
| `host/record-before.txt` | The live-registry test failing against an earlier commit's stale record (`expected 15, found 28`), and the CPU `ptxas` register counts for sm_80/86/89/120/121 (transcript; the raw output was not kept). |
| `host/tip-gates.txt` | At `c8dc633c` (this tip's tree): `zig build test test-golden fuzz-cortheon compile-cuda-tests -j1 --summary all`. |
| `host/tip-contract.txt` | At `c8dc633c`: `zig build test-kernel-contract -j1 --summary all` (the host-side contract tests, including the live-registry record test). |
| `host/tip-fmt.{txt,exit}` | `zig fmt --check` on the seven changed Zig files. |
| `host/lean-{base,tip,new}.txt` | `tools/zig/lean_check.py` on main and on `c8dc633c`, `tools/zig/lean_new.py` on the two. |
| `SHA256SUMS` | sha256 of every other file here. |
