# MiniMax-H3-Image on the Spark: mmh3 measured beside us, both engines on unseen prompts

- `docs/recipes/minimax-h3-cuda/side-by-side-job.sh` runs the two servers in one TOP window:
  `bench-script.sh` takes the five-run prompt list `fox, harbour, meadow, steam, fox` (run 5 a
  kept-states hit, labelled; runs 2-4 prompts neither server has seen), then `mmh3 server` answers
  the same list in the same order — seed 42, six distilled steps, 2048², mmh3's defaults. mmh3's
  per-run client walls and generation reports land in `mmh3-side-by-side/SUMMARY.md`.
- `bench-script.sh` learns `H3_BENCH_PROMPTS`, one prompt file a run (default: the pilot prompt
  four times, as before); `prompts/` holds the three unseen 2048² prompts.
- Measured on TOP's GB10 (job `top-d2fb1f45a475`, `0e2a3a5b`): mmh3's warm server took
  10.6/10.6/10.1 s on the unseen prompts and 10.1 s on the repeated fox (its INT8 encoder stays
  resident: an unseen prompt costs it 0.2 s), against our 15.1/15.2/15.1 s unseen and 13.9 s
  repeated — mmh3 is faster, not the card's ≤ 7 s. Every selftest passed; the same-latent
  `reference` compare held (56.31 dB, rel. 3.33e-3); vs episode 1's Mac picture the gate run is
  0.9952 cosine, 28.69 dB, 4.02 apart — all gates pass. The table and the two engines' remaining
  differences are in `docs/recipes/minimax-h3-cuda.md`.
