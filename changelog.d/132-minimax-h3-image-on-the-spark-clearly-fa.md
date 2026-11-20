# MiniMax-H3-Image on the Spark: mmh3 measured beside us, both engines on unseen prompts
- `docs/recipes/minimax-h3-cuda/side-by-side-job.sh` runs the two servers in one TOP window:
  `bench-script.sh` takes the five-run prompt list `fox, harbour, meadow, steam, fox` (run 5 a
  kept-states hit, labelled; runs 2-4 prompts neither server has seen), then `mmh3 server` answers
  the same list in the same order — seed 42, six distilled steps, 2048², mmh3's defaults. mmh3's
  per-run client walls and generation reports land in `mmh3-side-by-side/SUMMARY.md`.
- `bench-script.sh` learns `H3_BENCH_PROMPTS`, one prompt file a run (default: the pilot prompt
  four times, as before); `prompts/` holds the three unseen 2048² prompts.
- Measured on TOP's GB10 in one window: mmh3's warm server took 12.0-12.2 s on the unseen prompts
  (9.4 s on the repeated prompt, its states kept) against our 15.0 s unseen (13.9 s repeated) —
  mmh3 is faster on both measures, not the card's ≤ 7 s; the write-up in
  `docs/recipes/minimax-h3-cuda.md` has the per-run table, our selftests, the same-latent
  `reference` compare and the episode-1 Mac-picture gates, and says where mmh3 stays ahead.
