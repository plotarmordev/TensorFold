# MiniMax-H3-Image, fast and normal: one warm 2048² picture on the Spark and the Mac

The same request everywhere: a warm 2048 x 2048 picture, the pilot `prompt.txt`
("A red fox sitting in a snowy birch forest at dawn, soft golden light through the trees, highly detailed
photograph.", sha256 `cc969bcab79793c5035d2f0186e09a75efefe9feea34963052940925cd1af499`), seed 42, timed from
the request line to the saved PNG. Fast is `--draft-steps 3`: the first three of the six distilled steps on the
half-size canvas, then three at full size — a *different* picture, not a faster copy of the six-step one.

Warm means the resident servers already drew the same picture once (the Spark bench's runs 2-4 of 4, the Mac
run's requests 2-4 of 4); every number below is the warm mean of three. Both deployments keep the prompt's
states between repeats, so a warm row's prompt is ~0 s — the encoder and refiner are skipped entirely. These
are deployment numbers — TensorFold's CUDA path on TOP's GB10 against its Metal path on a Mac Studio's M5
Ultra — not a chip comparison: the two paths' stages are measured differently and the Mac's "drawing steps"
column includes the draft's lift inside its six-step count.

| Row | Prompt s | Drawing steps s | Decode s | Picture total s | Receipt |
|---|---:|---:|---:|---:|---|
| Spark normal | 0.0 | 12.9 | 1.3 | 14.2 | `top-30920108b62e` |
| Spark fast (`--draft-steps 3`) | 0.0 | 8.6 | 1.4 | 9.9 | `top-c2896139b5eb` |
| Mac normal | 0.0 | 9.2 | 0.8 | 10.0 | `h3-image-fast-and-normal/mac-normal.server.log.txt` |
| Mac fast (`--draft-steps 3`) | 0.0 | 6.1 | 0.9 | 7.0 | `h3-image-fast-and-normal/mac-drafts.server.log.txt` |

The Spark rows come from two `swarm gpu-run` jobs of
[`bench-script.sh`](recipes/minimax-h3-cuda/bench-script.sh) and
[`bench-drafts.sh`](recipes/minimax-h3-cuda/bench-drafts.sh) at `fa508cd6d` (the card's base, `origin/main`),
whose per-run numbers are in `h3-image-fast-and-normal/spark-normal-SUMMARY.md` and
`spark-drafts-SUMMARY.md`. The Mac rows are card 148's re-run of `tf-h3-image-server` (ReleaseFast, main at
`ef68ece55` with card 152's decoder) fed four request lines under `/tmp/tensorfold-model.lock`, the load average
3-7 with no other model running, every step `timing_valid=true`; timing lines and the full logs in the same folder. Each
server keeps the transformer bound across requests: "drawing steps" is the transformer's six steps (the three
half-canvas drafts plus the lift and three full steps on fast), and totals are the server's own prompt-to-PNG
line.

The same warm 15.5 → 10.7 s gap the Spark port measured in card 131 shows here at a better baseline: on main
the fast mode cuts each deployment's warm picture by a quarter to a third (Spark −4.3 s, Mac −3.0 s), all of it
in the drawing steps — three half-size steps at a quarter of the pixels are much cheaper than the three full-size
steps they replace. The Mac fast row's warm pictures took 6.88, 6.86 and 7.14 s (normal 9.97, 9.96, 9.98 s). An
earlier session at load 74-79 drew them in 7.4-8.5 s on the same 6.2 s of step GPU, so the fast row's wall moves
with CPU load; that session is replaced.

## Pictures

The fast mode draws a different picture on purpose (the half-canvas draft renoises the seed's detail), so the
pairs below are side-by-side looks, not pixel gates — composition and palette hold, fine detail changes.

| Spark (GB10, CUDA) | Mac (M5 Ultra, Metal) |
|---|---|
| ![Spark normal left, fast right](h3-image-fast-and-normal/spark-normal-vs-fast.png) | ![Mac normal left, fast right](h3-image-fast-and-normal/mac-normal-vs-fast.png) |
