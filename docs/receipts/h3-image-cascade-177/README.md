# Card 177: the resolution cascade against six full steps

The receipts behind [the metal recipe's card 177 section](../../recipes/minimax-h3-image-metal.md#card-177-the-resolution-cascade-cascadem-measured-against-six-full-steps),
measured on 22 November 2026, 09:39 to 09:45 (+0800), on an M5 Ultra.

| File | What |
| --- | --- |
| `session.sh` | One locked resident session per schedule, `session.sh none`, `2` or `4` (`--draft-steps`): `zsystem flock` on `/tmp/tensorfold-model.lock` before the server starts, released after it quits. |
| `text.txt`, `faces.txt`, `texture.txt`, `architecture.txt` | The fixed prompts, seed 7, 2048x2048; the warm-up request reuses `text.txt`. |
| `none.out`, `2.out`, `4.out` | Each server's own log: every step's GPU time and `timing_valid`, every request's stage times. |
| `loads.txt` | `vm.loadavg` before and after each request. |
| `gates.py`, `gates.txt` | Idea 1's picture gates, each cascaded PNG against the uncascaded one, run on the CPU in a venv with `torch`, `torchvision`, `lpips`, `pillow` and `numpy`. |
| `SHA256SUMS` | Each request's PNG and latent (the files themselves are not kept). |

The build was this card's ReleaseFast `tf-h3-image-server`, SHA-256
`e57755c5decef624846ec780ed12cd71a4491e16e9856b91ec4ab72b7ee661d8`.
