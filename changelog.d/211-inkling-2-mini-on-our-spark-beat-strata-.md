# Inkling 2 Mini on CUDA: a hill-climb script and its first GB10 receipt

- `docs/recipes/inkling-2-mini-cuda/hillclimb.sh` A/Bs decode levers with a control on each side, at 1 and 64
  streams, after the exactness gates, keeping a lever only past the 3% noise. `ceiling.py` gives the no-draft limit,
  and `setup.sh` holds the build and gates `run-script.sh` shares (now with 188's loop contract).
- On TOP's GB10 the default engine gives 32.6-32.9 tok/s at one stream and 590-597 total at 64, with no request
  failed and 0 of 32 tokens off the float64 reference. `TF_NVFP4_DECODE_WARPS=1` and `TF_QMMF_HALF_BELOW=0` stay
  inside the noise and are rejected.
- Strata v0.2.19's 74.3 tok/s at one stream is above this checkpoint's no-draft limit (49.2 tok/s at 273 GB/s), so
  passing it needs a CUDA DSpark drafter for the Mini first (card 215); the Strata bar moves to card 219.
