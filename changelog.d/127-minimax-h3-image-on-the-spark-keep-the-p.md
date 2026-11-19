# MiniMax-H3-Image on the Spark: keep the prompt's states across pictures

- `tf-h3-image-prompt-cuda --serve` keeps each prompt's states — its rows' ids and tags and the
  refiner's text rows — keyed by the prompt's exact bytes, least recently used inside a byte budget
  (`H3_PROMPT_CACHE` MiB, 256 by default; 0 keeps nothing). A repeated prompt runs no encoder or
  refiner kernels and writes the same case; an unseen prompt encodes as before.
- The key is the prompt bytes alone because the served process's tokenizer, marks and geometries are
  checked at load, and H3-Image has no negative prompt or CFG mode: nothing else changes the rows.
- `prompt_cache.zig` (`zig/src/families/h3`) holds the states; `prompt_cache_test.zig` gates a hit's
  verbatim rows, the LRU order inside the budget, the over-budget and zero-budget refusals and the
  already-kept no-op — `zig build test`.
- Measured on TOP's GB10 in one job (`docs/recipes/minimax-h3-cuda/bench-127-ab.sh`, patch -R for the
  main baseline), three runs each at 2048 x 2048 with `--tc`: repeated prompt 0.00 s (a kept-states
  hit, the encoder skipped — main re-encodes it in 1.05 s), unseen prompt 1.24 s (main 1.25 s, no
  slower); transformer 12.6 s and decoder 1.3 s unchanged, `repeat-3.png` byte-identical to the cold
  run's. Card 126's headline race uses the unseen-prompt figure.
