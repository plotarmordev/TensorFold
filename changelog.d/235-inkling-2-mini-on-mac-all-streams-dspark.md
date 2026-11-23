# Inkling 2 Mini on Metal: shared DSpark holds across streams

- Schedule greedy DSpark holds by block and recurrent step across active streams, batching the unembedding weight read for each cohort rather than completing one stream's draft chain before starting the next; a lone stream keeps its solo chain.
- Check greedy draft-token equality against solo holds with unequal depths, streams joining or leaving, and 16-stream cohorts.
- On the M5 Ultra, greedy DSpark reaches 317 / 260 tok/s at four streams (code / chat, up 14% on the better control) and 372 / 349 at sixteen (up 31%), with acceptance unchanged; `docs/receipts/inkling-2-mini-metal-235` holds the control/candidate/control medians and acceptance by depth.
