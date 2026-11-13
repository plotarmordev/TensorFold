### Ornith Metal: a lone MTP stream keeps drafting across served requests

- A lone MTP round whose draft was kept whole is kept inside the next draft call, and the held draft was dropped before that keep, so its landing went uncounted: only refused drafts trained the draft's landing chance, which fell until the lone window priced the draft out, and a server's later requests drafted nothing. The held draft now stays until that keep.
- On one stream, MTP now drafts every round of every request: 117.4 tok/s on a dedicated server and 117.3 at four capacity (plain 92.9), 75.9% of drafts accepted, same tokens as plain. The recipe's one-stream MTP rows are re-measured and lose their "drafting not sustained" label.
- `zig build test-qwen27-mtp-lanes` sends three consecutive lone requests through one engine and fails if any drafts in fewer than 90% of its rounds.
