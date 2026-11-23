# Strata Inkling 2 Mini GB10 baseline recipe

- Add a pinned v0.2.19 Strata Inkling/Control staging recipe that reuses the SQ-20 and SQ-10 packs already on TOP rather than uploading either pack again.
- Benchmark both packs at 1, 4 and 64 slots with Strata's own defaults and the concurrent HTTP client; record prompt content, KV mode, output length, medians, and the published SQ-10 64-slot gate without treating an unmatched rate as a bar.
- Run one pack at one stream count a GPU job (`strata-sq20-1.sh` through `strata-sq10-64.sh`); setup logs `build: N s` and only the measurement runs under `timeout 600`, logged as `measure: N s`.
