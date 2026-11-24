# Inkling 2 Mini on Metal: card 210's M5 Ultra speed work on main's engine

- Ports card 210's Metal speed work onto main's Inkling 2 Metal engine, commit by commit with `-x`: sethforprivacy's
  stack (routed experts a simdgroup an output and grouped by expert, on the tensor units and on turned 16×32 tiles for
  wide chunks; the forward on a concurrent encoder; logits and prompt projections on the tensor units; prompt chunks
  of up to 1,024 rows; the prompt kernel's register-held sums and in-cache tiles; the shorter convolution), gilby's
  DSpark lane state in the prompt cache (#4542), sethforprivacy's DSpark hold on the tensor units priced by the engine
  (#4660 with #4465, as one commit of that merge's net change), card 235's step-major greedy DSpark holds and their
  cohort launches, card 234's in-place verify window, and the float64 greedy gate with jschmied's `mini_ref` credited.
- Skipped because main already has them: the norm binding, a later stream's relative rows, the per-head snapshot copy
  (main keeps sethforprivacy's #4607 version over #4542's), the serving tests' first-token draw and the checkpoint
  file cache export. Card 248's one-submit candidate was reverted on its own branch and is not ported.
