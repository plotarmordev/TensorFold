# MiniMax-H3-Image on the Mac: step caching, measured and reverted

An opt-in `--step-cache X` on `tf-h3-image-server` reused the last run step's velocity for a skipped step, with a gate
on the latent's change deciding when. Skipping saves about 1.6 s a step at 2048, and `--step-cache 0` drew main's
bytes. But no gate kept card 80's picture rule. Reusing step 1's velocity for step 2 gave 15.9-20.6 dB, and skipping
step 6 alone 21.76 dB, against the 28.2 dB floor: each of the six distilled steps has its own head. The code is
reverted. The measurements are in `docs/recipes/minimax-h3-image-metal.md` under "Card 180", and "Next, ranked"
idea 5 now records the loss.
