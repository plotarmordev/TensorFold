# MiniMax-H3-Image on the Spark: the norms' AdaLN tables in bf16

- `h3m_project` writes the AdaLN modulation tables as bf16 instead of bf16-rounded values in f32 words, which halves `Inputs.tables`; `Inputs.tables` must be 8-byte aligned.
- `h3_norm_quant` and `h3_norm_quant_fp4` read each lane's four norm weights, scales and shifts as one 8-byte load. The fp32 sums and their order are unchanged, so int8/fp4 codes, scales and every picture stay byte-identical to main.
- Measured on GB10 (`bench-181-ab.sh`, job `top-d7a9943165b5`): `norms` 0.62 → 0.60–0.61 s; warm 2048² walls within noise of main. The recipe's card 181 section has the tables.
