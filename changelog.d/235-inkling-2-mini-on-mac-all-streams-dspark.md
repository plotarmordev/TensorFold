# Inkling 2 Mini on Metal: shared DSpark holds across streams

- Schedule greedy DSpark holds by block and recurrent step across active streams, batching the unembedding weight read for each cohort rather than completing one stream's draft chain before starting the next; a lone stream keeps its solo chain.
- Check greedy draft-token equality against solo holds with unequal depths, streams joining or leaving, and 16-stream cohorts.
- On the M5 Ultra, greedy DSpark reaches 310 / 257 tok/s at four streams (code / chat, up 14% / 37% on the better control) and 344 / 235 at sixteen, with acceptance unchanged; `docs/receipts/inkling-2-mini-metal-235` also holds the 1, 4 and 64-stream receipt against oMLX 0.8.2 on the same machine.
