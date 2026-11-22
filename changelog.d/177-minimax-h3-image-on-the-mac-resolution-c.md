# MiniMax-H3-Image on Metal: the resolution cascade measured

- `cascade=M` is `--draft-steps M` with no `--renoise-sigma`: M half-grid steps, `cascade.lift`, the rest at full size;
  without it the run is unchanged. A test pins the checkpoint's own sigmas, timesteps and deltas for every M.
- Warm 2048 on the M5 Ultra, four fixed prompts: the steps take 0.76x (M = 2) and 0.53x (M = 4) of six full ones.
  Idea 1's picture gates fail: LPIPS 0.41-0.73 against at most 0.08. The cascade stays opt-in. Receipts in
  `docs/receipts/h3-image-cascade-177`.
