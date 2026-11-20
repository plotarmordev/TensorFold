#!/bin/sh
# bench-script.sh twice in one GPU window: the picks file absent, then present. The first phase's own
# cold run writes it, so its second phase is not run (a present file's cold run is the next job's).
set -eu
model=/assets/drama3-pilot/models/MiniMax-H3-Image
pilot=/assets/drama3-pilot/h3-image-pilot
commit=$(git -c safe.directory='*' rev-parse HEAD)
bin=/work/out/tensorfold-cuda-$commit
out=/work/out/h3-image-cache-ab
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
H3_BUILD=sm121 sh docs/recipes/minimax-h3-cuda/cuda-build.sh
cachefile=$HOME/.cache/tensorfold/h3-cublaslt-algos.bin
rm -rf "$out" && mkdir -p "$out"
runbench() { # runbench OUTDIR: the bench's four 2048 runs into its own out, then its SUMMARY
    H3_BENCH_OUT="$1" sh docs/recipes/minimax-h3-cuda/bench-script.sh
}
{
    echo "commit: $commit"
    echo "GPU, driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
} >"$out/settings"
# A: every heuristic searched, the picks file written by the run itself; its host-compare is skipped
# (phase B's keeps the picture gate, and the latents are compared bit for bit below)

rm -f "$cachefile"
H3_BENCH_SKIP_COMPARE=1 runbench "$out/without-cache"
# B: the same binary with the picks file A left: cold and warm reads
grep -h 'h3 cublaslt picks\|retained at close' "$out/without-cache"/*.server.log 2>/dev/null | sort -u || true
ls -l "$cachefile" || { echo "the run wrote no picks file" >&2; exit 1; }
runbench "$out/with-cache"
grep -h 'h3 cublaslt picks' "$out/with-cache"/*.server.log 2>/dev/null | sort -u || true
if cmp -s "$out/without-cache/2048-1.latent.f32" "$out/with-cache/2048-1.latent.f32"; then
    echo "cold latents bit-identical" | tee "$out/latent-cmp.txt"
else
    echo "cold latents differ" | tee "$out/latent-cmp.txt"
fi
