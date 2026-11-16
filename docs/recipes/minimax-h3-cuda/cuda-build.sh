#!/bin/sh
# Builds this commit's CUDA kernel images and tf-cuda-test on TOP's Spark and audits the images' limits, through the
# Desk's GPU door (never `bottom shell` or `middle shell`: those Sparks serve DeepSeek):
#   ssh top-menu "job start gpu top plotarmordev/TensorFold SHA docs/recipes/minimax-h3-cuda/cuda-build.sh"
# The job has no network and no C++ compiler: zig-stage.sh puts Zig 0.17.0 in the pilot folder, and its clang is
# nvcc's host compiler. In /work/out:
# - tensorfold-fatbins-SHA: the sm_121 images and SHA256SUMS, for -Dfatbins (../../receipts/drama3-pruned-main/run-script.sh)
# - tensorfold-cuda-SHA: tf-cuda-test, with its images built in, and the five release SMs' images in fatbin/
# - limits.log: kernel-contract-limits over those images; the job exits with its status
# Later TOP jobs read either folder from the pilot folder once the Desk returns it there:
#   ssh top-menu "job fetch ID" | tar -x -C DIR
#   tar -C DIR/out -c tensorfold-fatbins-SHA | ssh top-menu "put top tensorfold-fatbins-SHA"
set -eu
zig=/assets/drama3-pilot/zig-aarch64-linux-0.17.0
sha=$(git -c safe.directory='*' rev-parse HEAD)
out=/work/out
tools=/work/tools
[ -x "$zig/zig" ] || { echo "missing $zig/zig: run zig-stage.sh; /assets holds:" >&2; find /assets -maxdepth 3 >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# the checkout may be read-only: caches and builds go to /work
export PATH=$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p "$tools"
# CUDA 13.0 does not list Zig's clang release among its host compilers
printf '#!/bin/sh\nexec %s c++ "$@"\n' "$zig/zig" >"$tools/clang++"
printf '#!/bin/sh\nexec /usr/local/cuda/bin/nvcc -ccbin %s -allow-unsupported-compiler "$@"\n' "$tools/clang++" >"$tools/nvcc"
chmod +x "$tools/clang++" "$tools/nvcc"
# -Dfatbins takes images holding the one SASS profile, so sm_121 is built apart from the audit's five SMs
zig build fatbins -Dsm=121 -Dnvcc="$tools/nvcc" --prefix /work/sm121
fatbins=$out/tensorfold-fatbins-$sha
mkdir -p "$fatbins"
cp /work/sm121/fatbin/*.fatbin "$fatbins"
(cd "$fatbins" && sha256sum *.fatbin >SHA256SUMS)
zig build install fatbins kernel-contract-fatbins -Dsm=80,86,89,120,121 -Dnvcc="$tools/nvcc" -Doptimize=ReleaseFast \
    --prefix /work/release
cuda=$out/tensorfold-cuda-$sha
mkdir -p "$cuda"
cp -r /work/release/bin/tf-cuda-test /work/release/fatbin "$cuda"
status=0
"$cuda/tf-cuda-test" kernel-contract-limits "$cuda/fatbin" >"$out/limits.log" 2>&1 || status=$?
tail -n 3 "$out/limits.log"
echo "kernel-contract-limits exit $status"
exit "$status"
