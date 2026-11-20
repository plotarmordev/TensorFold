# MiniMax-H3-Image on the Mac: keep the prompt's states across pictures

- `tf-h3-image-server` keeps each reference-free prompt's states — its rows' ids and tags and the
  refiner's text rows — keyed by the prompt's exact bytes, least recently used inside a byte budget
  (`H3_PROMPT_CACHE` MiB, 256 by default; 0 keeps nothing). A repeated prompt runs no encoder or
  refiner kernels and writes the same case; an unseen prompt encodes as before, and an edit's
  `--reference` case always does: its vision rows sit inside the ids.
- The cache is card 127's `prompt_cache.zig` (`zig/src/families/h3`), re-exported from `prompt.zig`
  and held by `h3_image_prompt.zig`'s `Encoder`; the key is the prompt bytes alone because the
  tokenizer, marks and geometry are checked at load and H3-Image has no negative prompt or CFG.
- Warm 2048 A/B on the M5 Ultra, one resident server per build, card 34's `prompt.txt` cold then twice
  warm and three unseen prompts: repeated prompt 0.00 s (a kept-states hit — main re-encodes it in
  2.75-3.50 s), unseen prompt 0.22 s (main's cold encode 0.32 s), 6 steps and decoder unchanged, and
  the warm PNGs byte-identical to the cold run's and main's (SHA-256 `005f01f3`).
- `prompt_cache_test.zig`'s gates — a hit's verbatim rows, LRU inside the budget, the over-budget
  and zero-budget refusals — now run on the host under `test-h3-image-prompt` too.
