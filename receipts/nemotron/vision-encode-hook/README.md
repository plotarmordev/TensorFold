# Receipts: an encode hook on `Vision`, and a host test that a request encodes each image once

Branch tip `592610353b1d0b2e16a069a0f22e28ff35e56ae1` on main `d1018b318fb51981c4a18f12e40269a4d3ee9713`, two commits:
`patches/0001` is #1057's change rebased onto main (`git patch-id --stable` equal to #1057's head `e2357b58`), `patches/0002`
the hook and the test. CPU only, one DGX Spark (GB10, aarch64 Linux), zig 0.17.0, nvcc 13.0, `-j1`, each step on a fresh
cache. Paths private to the machine are placeholders (`<checkout>`, `<out>`, `<zig>`, `<cache>`).

| Path | What |
|---|---|
| `scripts/gates.sh` | The script that ran every step below, as run; `host/gates.txt` is its output. |
| `host/mutant.new-test.log` | The new test alone (`zig build test-modelopt-host -Dmodelopt-host-filter=<its name>`) on the tip with `0001` reverted, which is main's prompt pass with the hook: `expected 1, found 2`. The reverted diff is in `host/gates.txt`. |
| `host/tip.new-test.log` | The same step on the tip: passes. |
| `host/tip.{test,test-kernel-contract,test-golden,fuzz-cortheon}.log` | Upstream's CI steps on the tip. |
| `host/base.test.log` | `zig build test` on main. |
| `host/{base,tip}.lean.txt`, `host/lean-new.txt` | `tools/zig/lean_check.py` on main and the tip; `tools/zig/lean_new.py` between them. |
| `host/tip.{fatbins,native}.log` | `zig build fatbins -Dsm=121` and `zig build native -Doptimize=fast -Dstrip=true` on the tip. |
| `patches/` | `git format-patch main..tip`. |
| `SHA256SUMS` | sha256 of every other file here. |
