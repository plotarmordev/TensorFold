# DeepSeek-V4.1 serve steps: the decode-between-pieces test no longer races the wall clock

- `serve_steps_test.zig`'s first case submitted its decoding reply, waited 1 ms at a time for one decode round, then submitted the 700-row prompt and slept 1 ms a piece. On a fast run the reply's 400 rounds could all finish inside that wait, so nothing decoded while the prompt filled and the finish-window assertion saw no round.
- It now submits both requests back to back, and a gate on the first finish holds the host's thread until both are queued, so no round can run before the reply the long prompt must interleave with. The short prompt fills whole and decodes first; the long prompt then takes one 16-row piece a host visit, the reply's round after each.
- The assertion counts decode rounds between two consecutive pieces (`most_between > 0`, a scheduler counter) instead of rounds inside a begin-to-finish window, and the reply is 64 tokens instead of 400, so the test sleeps nowhere. Filling every piece in one visit leaves that counter at zero and fails it.
- No serving code changed.
