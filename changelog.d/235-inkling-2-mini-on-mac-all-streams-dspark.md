# Inkling 2 Mini on Metal: shared DSpark holds across streams

- Schedule greedy DSpark holds by block and recurrent step across active streams, batching the unembedding weight read for each cohort rather than completing one stream's draft chain before starting the next.
- Check greedy draft-token equality against solo holds with unequal depths, streams joining or leaving, and 16-stream cohorts.
