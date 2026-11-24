# Inkling 2 Mini on Metal, M5 Ultra: card 261's receipt

Apple M5 Ultra, macOS 27.0; the official checkpoint (`Inkling-2-Mini`, its own `mtp.safetensors` as the DSpark
drafter). Card 261 ports card 210's Metal speed work onto main's Inkling 2 Metal engine (see
`changelog.d/261-inkling-2-mini-on-mac-card-210-s-m5-ultr.md` for what came over and what main already had).

## Ceiling

`blit.swift` (card 210's, unchanged), a device-to-device blit of 4 GiB, best of 12, under
`/tmp/tensorfold-model.lock` inside the Desk's Mac GPU window at 713d0d7ff: **1,080 GB/s** (read plus write). Card 210
measured 1,072 on the same Mac. With its 5.55 GB a plain token and 10.41 GB for four, the plain ceilings stay at about
195 tok/s for one stream and 415 tok/s for four.

## Exactness

`TF_INKLING2_METAL_DIR=$HOME/models/Inkling-2-Mini zig build test-inkling2-metal-float64` at 713d0d7ff passes on the
official checkpoint: 0 of 32 greedy tokens differ from the float64 host reference, and the planted control without
cross-pass messages disagrees (6 of 6 tests, 11 min, the device part under the model lock).

## Speed

Not measured yet. The window closed during the first control run (main, plain, one stream), which decoded at
0.8 tok/s; that run is void and no figure here is compared with #4618's published 123 / 281 tok/s. `bench.sh` holds
the run: control (main) / candidate / control, each plain and with `--drafter`, at 1 and 4 streams, each under
`timeout 600` and the model lock.
