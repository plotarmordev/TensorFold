# Sourced by setup.sh and round64.sh: finds the checkpoint, unpacks the staged toolchain and builds sm_121 into
# /work/sm121. Sets $model, $bin and $out; any failure stops the job.
pilot=/assets/drama3-pilot
staged=$pilot/cuda-build-toolchain
zig=zig-aarch64-linux-0.17.0
gcc=arm-gnu-toolchain-14.3.rel1-aarch64-aarch64-none-linux-gnu
out=/work/out/inkling2
model=
for dir in "$pilot/Inkling-2-Mini" "$pilot/models/Inkling-2-Mini"; do
    [ -f "$dir/config.json" ] && model=$dir && break
done
if [ -z "$model" ]; then
    echo "no Inkling-2-Mini/config.json in $pilot or $pilot/models; Inkling configs under /assets:" >&2
    find /assets -maxdepth 5 -name config.json -exec grep -l '"inkling2' {} + >&2 || true
    exit 1
fi
[ -d "$staged" ] || { echo "missing $staged: run ../minimax-h3-cuda/toolchain-stage.sh" >&2; exit 1; }
(cd "$staged" && sha256sum -c --quiet) <<EOF
9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $zig.tar.xz
7f4ca1d3b773c41b031019d1249b2c58e6beb470eeabacfa0a0813c5e8bb7e5b  $gcc.tar.xz
EOF
tar -C /work -xJf "$staged/$zig.tar.xz"
tar -C /work -xJf "$staged/$gcc.tar.xz"
# the checkout may be read-only: caches and builds go to /work
export PATH=/work/$zig:$PATH ZIG_GLOBAL_CACHE_DIR=/work/zig-cache ZIG_LOCAL_CACHE_DIR=/work/zig-cache
export LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NVCC_PREPEND_FLAGS="-ccbin /work/$gcc/bin/aarch64-none-linux-gnu-g++"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
echo "model $model: $(cat "$model"/*.safetensors | sha256sum | cut -d' ' -f1)"
zig build install native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/sm121
bin=/work/sm121/bin
rm -rf "$out"
mkdir -p "$out"
