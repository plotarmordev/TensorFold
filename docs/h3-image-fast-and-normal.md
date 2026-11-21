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
| Mac fast (`--draft-steps 3`) | 0.0 | 6.1 | 0.7 | 6.8 | `h3-image-fast-and-normal/mac-drafts.server.log.txt` |
| Spark normal, card 149, same prompt again | 0.0 | 10.8 | 1.2 | 12.1 | `top-a84c94b49c92` (one run) |
| Spark normal, card 149, unseen prompts | 2.0 | 10.9 | 1.2 | 14.0 | `top-a84c94b49c92` (three runs) |
| mmh3 server, same job, unseen prompts | 0.2 | ~9.5 | 0.6 | 10.6 | `top-a84c94b49c92` (three runs) |

The Spark rows come from two `swarm gpu-run` jobs of
[`bench-script.sh`](recipes/minimax-h3-cuda/bench-script.sh) and
[`bench-drafts.sh`](recipes/minimax-h3-cuda/bench-drafts.sh) at `fa508cd6d` (the card's base, `origin/main`),
whose per-run numbers are in `h3-image-fast-and-normal/spark-normal-SUMMARY.md` and
`spark-drafts-SUMMARY.md`. The Mac rows are card 148's head (ReleaseFast; its render path is main's, `d9d87fe67`,
whose sessions beside it in the same folder drew the same bytes in the same times) fed four request lines under
`/tmp/tensorfold-model.lock`, the load average 5-7 (4.9-7.1) with no other model running, every step `timing_valid=true`. Each
server keeps the transformer bound across requests: "drawing steps" is the transformer's six steps (the three
half-canvas drafts plus the lift and three full steps on fast), and totals are the server's own prompt-to-PNG
line.

The same warm 15.5 → 10.7 s gap the Spark port measured in card 131 shows here at a better baseline: on main
the fast mode cuts each deployment's warm picture by a quarter to a third (Spark −4.3 s, Mac −3.1 s), all of it
in the drawing steps — three half-size steps at a quarter of the pixels are much cheaper than the three full-size
steps they replace. The Mac fast row's warm pictures took 6.82, 6.83 and 6.83 s (normal 9.96 s each). Earlier
sessions at load 74-79 drew them in 7.4-8.5 s on the same 6.1 s of step GPU, so the fast row's wall moves with CPU
load; those sessions are replaced.

Card 149 added the last three rows, measured at `99c900f5` once the encoder, decoder, passes and int8-product cards
had merged. Its runs are not this table's warm repeats. The first row is a single repeat of the fox after it was
drawn cold. The other two are means over three prompts the server had never seen, and mmh3's server ran in the
same TOP job. The rows lower the Spark's warm normal picture from 14.2 to 12.1 s. On unseen prompts it takes 14.0 s
against mmh3's 10.6 s, so mmh3 is still 3.4 s ahead. Every run and the stage breakdown are in
[the CUDA recipe's card 149 section](recipes/minimax-h3-cuda.md#card-149-the-race-again-after-the-encoder-decoder-passes-and-int8-product-cards-21-november).

## Pictures

The fast mode draws a different picture on purpose (the half-canvas draft renoises the seed's detail), so the
pairs below are side-by-side looks, not pixel gates — composition and palette hold, fine detail changes.

| Spark (GB10, CUDA) | Mac (M5 Ultra, Metal) |
|---|---|
| ![Spark normal left, fast right](h3-image-fast-and-normal/spark-normal-vs-fast.png) | ![Mac normal left, fast right](h3-image-fast-and-normal/mac-normal-vs-fast.png) |
