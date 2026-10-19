# Receipts: a CUDA prompt pass with images runs the RADIO tower once

Branch tip `e2357b582f041a0352a8f34f97a2f1e0c7150bfc` on main `d46d17a784758b8a8b66badde1f728f36bdc9541`.

The GPU runs predate main's three latest commits: they compare TensorFold 1.1.0 (`76aafcf274e60a5e31a59ce145b2f71f7b366e9d`,
"base") with 1.1.0 plus this fix (`b4495aa3410294555e11f7fc621c2e0cd1ffd689`, "fix"). Nothing under `zig/`, `tools/`
or `build.zig` changed between 1.1.0 and main (`git diff --stat 76aafcf2 d46d17a7 -- zig build.zig build.zig.zon tools`
is empty), and the branch's diff against main is byte-identical to `git diff 76aafcf2 b4495aa3`.

Both GPU builds came from clean detached checkouts with `zig build fatbins -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline
-Dnvcc=<nvcc> -Dsm=121` and `zig build native -Dtarget=aarch64-linux-gnu.2.28 -Dcpu=baseline -Doptimize=fast -Dstrip=true
-Dversion=1.1.0 -Dfatbins=<out>/fatbin` (zig 0.17.0, nvcc 13.0.r13.0). `tensorfold-native` sha256: base
`04167a172c847952737a4b401fb5945b74d598a51312ed19f4e6ddd974015088`, fix
`52b3990e6f43b9711c26671fce7f0bb8c46f4f85bba5161cd48fab3da15853fd`; each hold's `.sha256` and `.hold.log` repeat it.

One DGX Spark (GB10, sm_121), driver 580.173.02, CUDA 13.0, checkpoint
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@82d93699acd3be9e4617c53360fe5b3f5dbeb1b3`. Every hold is a fresh
server in a fresh `HOME`: `tensorfold-native serve <checkpoint> --host <loopback> --port 8897 --name
nemotron-3.5-super-vl --vision --vision-image-tokens 13312 --vision-max-images 4 --context 18432 --parallel 1
--prompt-cache-gib 0` with `TF_HEAT_HIGH=85 TF_HEAT_LOW=75`; `.guard.log` samples GPU and zone temperature, load and
free memory every 5 s. The client is `scripts/bench.py image-ttft` (with `scripts/prompts.py`, both as run): one
generated 2048 x 1664 PNG, then four copies of it, each a warmup and 3 measured requests, greedy, thinking off,
`"draft": false`, 32 tokens, cooled to 75 °C before every request. One image is a 3,356-token prompt, four are 13,366.
Paths private to the machine are placeholders (`<home>`, `<out>`, `<checkpoint>`, `<loopback>`, `<node>`,
`<checkout>`, `<zig>`).

| Path | What |
|---|---|
| `gpu/tower-{base,fix}.*` | Each server under `nsys profile --trace=cuda --sample=none --cpuctxsw=none`; `*_cuda_gpu_kern_sum.csv` is `nsys stats --report cuda_gpu_kern_sum`. The 8 requests carry 20 images; `tf_vit_position_add` and `tf_vit_unshuffle` run once per image the tower encodes. |
| `gpu/ttft-{base-1,fix,base-2}.*` | Unprofiled, in that order, each hold started at a zone of 60 °C or less and a load under 2. |
| `host/` | Upstream's CI steps, run on CPU by `scripts/gates.sh` (`gates.txt` is its output): `tip.*.log` the tip's `test-kernel-contract`, `test`, `test-golden`, `fuzz-cortheon` and release build; `base.test.log` main's `zig build test`; `{base,tip}.lean.txt` and `lean-new.txt` the lean check. |
| `SHA256SUMS` | sha256 of every other file here. |
