# Inkling 2 Mini on CUDA: a hill-climb script and decode ceilings for TOP's GB10

- `docs/recipes/inkling-2-mini-cuda/hillclimb.sh` A/Bs decode levers with a control on each side of each lever, at
  1 and 64 streams, after the exactness gates. `ceiling.py` gives the bandwidth ceiling at N streams, and
  `setup.sh` holds the build and gates `run-script.sh` shares.
- Without drafts, the GB10 ceilings (49.2 tok/s at one stream, 534.8 total at 64) sit under Strata v0.2.19's 74.3 and
  1,145.1. Beating them needs a CUDA DSpark drafter for the Mini (the Metal one exists); no lever is measured yet.
