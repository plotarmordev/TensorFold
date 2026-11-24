# Inkling 2 Mini on Metal: a DSpark round's keep, absorb and hold share one submission
- Keep stages its blits and advances cache bookkeeping; the following draft stages the accepted-row absorb and
  greedy hold. The next readback encodes all three in one command buffer, commits once and waits once instead of
  three submissions and waits. Verify still runs separately: its sampled tokens and accepted path need the host.
- Prompt forwards and snapshot saves drain staged work before touching the shared taps or cache histories. Released
  streams drop their staged buffers. Sampled holds still read each pick between steps.
