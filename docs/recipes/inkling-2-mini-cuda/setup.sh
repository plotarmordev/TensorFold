# Sourced by run-script.sh and hillclimb.sh: finds the checkpoint, unpacks the staged toolchain, builds sm_121 into
# /work/sm121 and runs the exactness gates (#4387's serving checks, 32 greedy tokens against the float64 reference
# and its planted control). Sets $model, $bin and $out; any failure stops the job.
. docs/recipes/inkling-2-mini-cuda/toolchain.sh
echo "model $model: $(cat "$model"/*.safetensors | sha256sum | cut -d' ' -f1)"
zig build install native -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast --prefix /work/sm121
"$bin/tf-cuda-test" kernel-contract "inkling2 loop"
# #4387's serving checks take any 4,096+ ids; these are fixed so every run sees the same prompt
python3 -c 'import random; r = random.Random(4387); print(" ".join(str(r.randrange(1000, 199000)) for _ in range(4200)))' >"$out/ids.txt"
"$bin/tf-cuda-test" inkling2-serving "$model" "$out/ids.txt"
text="The three laws of thermodynamics, explained for a curious student:"
"$bin/tf-cuda-test" inkling2-mini-ref "$model" 32 "$text"
INKLING2_REF_PLANT=1 "$bin/tf-cuda-test" inkling2-mini-ref "$model" 32 "$text"
