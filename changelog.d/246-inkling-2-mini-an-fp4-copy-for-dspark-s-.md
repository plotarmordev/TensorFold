# Inkling 2 Mini DSpark: a private FP4 copy for the draft head's full-vocabulary products
- DSpark's unembed base and folded Markov Phi read a copy packed at load: E2M1 codes with one BF16 scale per 32
  columns, 4.5 bits a weight. A seven-row hold reads 261 + 29 MB instead of 926 + 103 MB. `INKLING2_DRAFT_HEAD`
  picks `fp4` (the default), `fp8` (E4M3) or `bf16` (the old path). The target's unembed and embed are hashed before
  and after the drafter loads and stay byte-identical. Greedy drafted replies equal the draft-disabled ones in the
  device suite and in every bench cell.
- M5 Ultra (`docs/receipts/inkling-2-mini-dspark-fp4-246`), accepted drafts on code / chat: BF16 180 / 149, FP8
  177 / 149, FP4 180 / 146 (at most 2.0% fewer). At four streams FP4 reaches 276.5 / 215.0 tok/s against BF16's
  242.3 / 171.1, with rounds 38.4 ms against 41.4 ms. One stream does not beat BF16 in these runs: 109.3 against
  116.4 on code. #4618's published 123 / 281 is still not beaten, and no other engine is installed on this Mac to
  measure beside it.
