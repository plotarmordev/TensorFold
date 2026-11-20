# MiniMax-H3-Image: one table, fast and normal mode on the Spark and the Mac

- `tf-h3-image-server` ran no request on main: card 138's `--draft-steps`/`--renoise-sigma` parsing collected the two positional paths and then read them as `args[1]`/`args[2]`, so the encoder loaded `MODEL_DIR/config.json` (missing) and every start failed. Positional zero-based now: `args[0]` is the text encoder, `args[1]` the model folder.
- Warm 2048x2048 pictures, seed 42 and the same prompt, timed prompt-to-PNG on main plus this fix on the Mac: Spark normal and `--draft-steps 3`, Mac normal and `--draft-steps 3`, each split into prompt, drawing steps and decode, with the TOP job ids and Mac run logs in `docs/h3-image-fast-and-normal.md` beside the fast-vs-normal side-by-sides.
