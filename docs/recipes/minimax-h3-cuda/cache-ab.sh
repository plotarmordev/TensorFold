#!/bin/sh
# bench-script.sh twice in one GPU window: the picks file absent (the first cold run writes it), then
# present. The two cold latents are compared bit for bit at the end; a difference fails the job.
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
runbench() { # runbench OUTDIR [VAR=value...]: the bench's four 2048 runs into its own out
    out_dir=$1 && shift
    env "$@" H3_BENCH_OUT="$out_dir" sh docs/recipes/minimax-h3-cuda/bench-script.sh
}
{
    echo "commit: $commit"
    echo "GPU, driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
} >"$out/settings"
# A: every heuristic searched, the picks file written by the run itself; its host-compare is skipped
# (phase B's keeps the picture gate, and the latents are compared bit for bit below)
rm -f "$cachefile"
runbench "$out/without-cache" H3_BENCH_SKIP_COMPARE=1
# B: the same binary with the picks file A left: cold and warm reads
grep -h 'h3 cublaslt picks\|retained at close' "$out/without-cache"/*.server.log 2>/dev/null | sort -u || true
# a header alone is 272 bytes (magic, key, count): phase B needs records to read
[ "$(wc -c <"$cachefile")" -gt 272 ] || { echo "the run wrote no picks" >&2; exit 1; }
ls -l "$cachefile"
runbench "$out/with-cache"
grep -h 'h3 cublaslt picks' "$out/with-cache"/*.server.log 2>/dev/null | sort -u || true
if cmp -s "$out/without-cache/2048-1.latent.f32" "$out/with-cache/2048-1.latent.f32"; then
    echo "cold latents bit-identical" | tee "$out/latent-cmp.txt"
else
    echo "cold latents differ" | tee "$out/latent-cmp.txt"
    exit 1
fi
