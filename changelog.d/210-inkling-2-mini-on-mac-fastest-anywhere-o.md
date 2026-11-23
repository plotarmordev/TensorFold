# Inkling 2 Mini on Metal: the open M5 work merged onto main

- Takes sethforprivacy's Metal stack from ashhart/TensorFold: the #4689 head, which carries #4446, #4448, #4523, #4608, #4618, #4631, #4644, #4645, #4649 and #4666 (tensor-unit experts and prompt projections, 1,024-row prompt chunks with direct key tiles, one-pass round logits, fused convolution steps); #4660 with #4465 (the DSpark hold in one command buffer on the tensor units, drafts priced by the engine); #4607 (a snapshot copies each key-value head's live rows); #4622 (the window fit prices a drafted token at a token's bytes); #4635 (grammar masks step by the padded vocabulary).
- Takes gilby's #4542 for its DSpark lane state in the prompt cache; its key-value copy is the same fix as #4607, whose layout is kept.
- Leaves out gilby's #4543, #4683 and #4685 and the EXL3 packs in #4471: each rewrites the same kernels, plans and contracts as the stack in parallel (87 and 16 conflicting files), and #4471 serves another checkpoint format, not the official one the bar is measured on.
- Not yet measured on this branch: the M5 Ultra speed and exactness runs need a Mac GPU window.
