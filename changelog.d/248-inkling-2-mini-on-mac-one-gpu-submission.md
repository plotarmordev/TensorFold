# Inkling 2 Mini on Metal: a one-stream DSpark round profiled; one submission a round measured no faster
- Folding keep, absorb and the greedy hold into one submission saved two commit-and-waits (14 µs each) of a 27 ms
  round and read 113.4 tok/s against controls 118.5 / 120.9 on the M5 Ultra, so the change is not kept.
- `docs/receipts/inkling-2-mini-metal-248/profile.md` splits the round (verify 60%, draft 40%, host waits 0.1%) and
  names the target forward's barrier chain, 3× its bandwidth floor, as the next lever.
