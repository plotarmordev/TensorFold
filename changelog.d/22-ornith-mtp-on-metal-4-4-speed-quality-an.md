### Ornith Metal: measured draft modes and reference quality

- Update the 122B recipe with plain, DFlash2 and MTP throughput, first-token latency and accepted/proposed draft counts at one and four streams, including the measured build, settings and equality receipts.
- Keep DFlash2 as the fastest measured default: 236.5 tok/s on a dedicated one-stream server and 257.3 aggregate tok/s with four streams. Metal selects MTP instead of DFlash2; they do not combine.
- Record MTP's lone-stream zero-proposal limitation rather than labeling fallback throughput as sustained MTP speed. Active four-stream MTP remains token-exact against plain decoding across all 16 quality prompts.
- Reproduce mlx-lm teacher-forced quality: native mean NLL 0.1126066 versus the reference's 0.1186612, with unchanged 636/2,048 leading-token matches.
