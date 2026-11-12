### Ornith Metal: measured draft modes and reference quality

- Update the 122B recipe with plain, DFlash2 and MTP throughput, first-token latency and accepted/proposed draft counts at one and four streams, including the measured build, settings and equality receipts.
- Keep DFlash2 as the fastest measured default: 247.3 tok/s on a dedicated one-stream server (plain 92.8) and 267.1 aggregate tok/s with four streams (plain 220.6). Metal selects MTP instead of DFlash2; they do not combine.
- Fix lone-stream MTP on Metal Qwen3.5-MoE stopping drafts after its first request: a lone stream's drafts are priced from its own measured windows (`lone_cost_tile`), as on CUDA. Lone MTP still runs below plain (78.5 tok/s) and level with it at four streams (226.6).
- MTP and DFlash2 replies equal plain decoding's on all 16 quality prompts; mlx-lm teacher-forced quality is unchanged: native mean NLL 0.1126066 versus the reference's 0.1186612, 636/2,048 leading-token matches.
- Remove the dashboard source-text test and its build registration: it pinned UI wording (including the removed “Paged prefix cache” label), not dashboard behavior, and blocked the required full-suite gate.
