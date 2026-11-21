# Card 152: the clean fast-mode A/B

These are the receipts behind the clean fast-mode rerun in
[minimax-h3-image-metal.md](../../recipes/minimax-h3-image-metal.md#card-152-the-decoder-on-the-tensor-units),
measured on 20 November 2026, 18:34 to 18:37 (+0800), on an M5 Ultra.

| File | What |
| --- | --- |
| `session.sh` | One locked fast session per build (`--draft-steps 3`): `zsystem flock` on `/tmp/tensorfold-model.lock` before the server starts, released after it quits. Run as `session.sh main head`. |
| `pilot.txt`, `v1.txt` to `v3.txt` | The prompts: the seed-42 pilot, then seeds 31-33. |
| `main.out`, `head.out` | Each server's own log: its load line and every request's stage times. |
| `loads.txt` | `vm.loadavg` before and after each request. |
| `SHA256SUMS` | Each request's PNG and latent (the files themselves are not kept). |

The builds were ReleaseFast `tf-h3-image-server`: main `4f805250e` (SHA-256
`1f4b1dabf3657309db3c3e3835479f847c54fe6114a262f2289d3cbeda0842af`) and this card's head (SHA-256
`2a36d2d375297f4e57c412fc66e5410ada4e145735ec052ba6a6080bfdfca9e5`), built before the gemm's transposed weights
moved to `frag_get_t`. On an M5 `nax.h` defines `frag_get_t` as `frag_get`, so the M5 build's text is unchanged.
`flock` does not write to the lock file, so its modification time does not change.
