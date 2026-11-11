# MiniMax-H3-Image speed, round 1: three independent sweeps, ranked

This round looked for ways to make one MiniMax-H3-Image picture at least 1.5x faster end to end on an M5 Ultra
(256 GB) running TensorFold's native Metal port. Three sweeps worked apart, each on a different model family, and
none saw another's proposals:

| Sweep | Question | Model family |
|---|---|---|
| Architecture | Where one picture's time goes in the text encoder, transformer and decoder, and what the maths allows | Claude Fable 5.2 |
| History | What our own past speed rounds (GLM, Nemotron, DeepSeek, Drama, Ornith, the H3 video DiT) teach an image model | GPT-6 Astra |
| Outside | Papers and other runners, as raw material, for what nobody has combined or run on a Mac | GPT-6 Astra |

The two GPT-6 Astra sweeps ran as separate sessions with separate briefs. Nothing below was run on a GPU. Agreement
between sweeps is the confidence signal; a lone idea needs its diagnostic first.

## What this was made on

- **TensorFold 1.2.7**, our main at `eeeb1ae3e` (card 43's catch-up with the maintainer), merged into this branch.
- **The H3-Image implementation analysed** is the maintainer's unmerged pull requests, at these heads: #2766
  transformer `9692b83d5331`, #2779 prompt path `0fa460a1c602`, #2780 decoder `9eb514c504ca`, #2781 end to end
  `fa0e45c77b68`, #2775 VAE host rules `f5dad8bcac70`, #2778 edit encoder `dd4ec5e3c017`, #2796 prompt cache
  `e782e6f20f30`, #2808 kernel contract `8e31db228219`, #2810 edits `32e520d08f58`.
- **Measured, inherited:** the M5 Ultra GEMM rates below (Qwen-Image 2.1, FastH3) and h3-image-mlx's 356.84 s on an
  M5 Pro, all from other runs. **Estimated:** every stage time, gain and cost in this report, until card 34 and the
  build cards replace them.

## What a picture costs, by the architecture sweep's model

Calibrated from two M5 Ultra receipts in this repository that use the same tensor-unit GEMM (Qwen-Image 2.1 at
89 TFLOPS, FastH3 at 76 TFLOPS), at 80 TFLOPS:

| Stage | 2048 x 2048 | 4096 x 4096 | Bound by |
|---|---|---|---|
| Three tool launches, each compiling its Metal library from source | 1-2 s | same | CPU |
| Text encoder: page-in and one pass | 1.5-3 s | same | page-in |
| Transformer load: map the checkpoint, build AdaLN tables, wire step weights | 3-6 s | same | page-in |
| Six steps | 13.9 s (2.31 s a step) | 76.6 s (12.8 s a step) | tensor units |
| of which linear layers | 12.0 s | 47.5 s | compute, cache-served |
| of which attention | 1.9 s | 29.1 s | compute and `exp` |
| Decoder | 1.0-1.6 s | 3.5-5.9 s | compute |
| Host tail: blend, RGB, single-thread PNG | 0.3-0.6 s | 1-2.5 s | CPU |
| **Total** | **about 23 s, 39% outside the steps** | **about 91 s, 16% outside the steps** | |

Facts the proposals rest on, checked in the port's code:

- **Attention is dense both ways.** Text rows query picture keys and take each step's timestep line, so no text K/V
  is constant across steps or blocks. Any cross-step reuse is approximate.
- **AdaLN tables are rebuilt on every launch.** Building them reads the 26 GB of `adaln_proj` weights to make 58 MB
  of tables. That explains the checkpoint's 66.3 GB against the 34.6 GB that 17.3 B bf16 block parameters need.
- **The linear layers are not DRAM-bound.** One 64 x 128 x 256 GEMM geometry serves all four projection classes;
  the SwiGLU's `wide` buffer is written and read back once a block.
- **The weights are near full rank** (block 25 needs rank 4,343-5,189 of 5,376 for 99% of the energy), and the six
  step heads differ by only 1.5-2.3% from each other.
- **The picture lattice is oversampled in position.** At 4096 the largest rotary phase between neighbouring rows is
  0.25 rad, and a 2 x 2 mean keeps at least 99.2% of the top frequency's magnitude.
- **The M5 has int8 tensor products**, but the external int8 run gives no better rate per GPU core than bf16 here,
  and memory is not the limit at 256 GB.

## Guardrails every proposal meets

- **Not already someone's.** Card 40's design (commit `bcc2f15c`: the resolution cascade, routed attention in the
  last steps, the decoder in 512-pixel tensor-unit tiles), the maintainer's open H3-Image pull requests (#2766,
  #2775, #2778-#2781, #2796, #2808, #2810) and h3-image-mlx (int8 W8A8, separate processes, resident or streamed
  weights, precomputed AdaLN, batched decoder tiles) count only where a proposal clearly goes beyond them, said in
  its row. So do the CUDA runners' graphs, compile, fused QK-norm and rotary, conditioning caches and FP8 repacks.
- **No rejected lesson returns without new evidence, named.** See [past results that bind](#past-results-that-bind).
- **Precision.** Upstream takes no lossy weight or cache formats and no output-changing mode by default. An exact
  proposal keeps every bf16 rounding point and passes the float64 kernel contract; a lossy one is opt-in, off by
  default with a byte-identical picture, and carries a picture-quality guard.
- **Each proposal has an A/B plan, a speed gate, a quality guard, an expected gain and a cost.**

### Shared A/B protocol

- **Held fixed:** checkpoint revision, prompt bytes, seed noise, the six sigmas and heads, decoder and precision
  outside the changed path.
- **Corpus:** 24 prompts fixed before any run (small text in Latin and Chinese, counting and spatial relations,
  faces and hands, thin structures, repeated textures, flat graphics, low-contrast scenes) x seeds {7, 42}, at
  2048 x 2048, 4096 x 4096 and one non-square canvas. Any tuning uses ten separate calibration prompts.
- **Runs:** the same binary with the change off and on, alternated A-B-B-A in one session with no other GPU work;
  cold and warm page cache reported apart.
- **Recorded:** wall time to the PNG, each stage's time, each step's GPU time, peak physical footprint.

### Speed gates

- **Local:** against card 34's stage times on this M5 Ultra (prompt, each transformer step, decode, load and glue).
  A 1.5x claim needs the lower 95% bound of the paired end-to-end speed-up at 1.5x or better.
- **External:** h3-image-mlx's teapot prompt, seed 42, at 4096 x 4096: 356.84 s on an M5 Pro (343.6 s of steps,
  8.37 s of decode). That is a different machine, so it is not an algorithmic comparison: card 3 below reruns
  h3-image-mlx on this M5 Ultra, and the round's cards together must be at least 3x faster than that rerun.

### Quality guards

- **Exact:** byte-identical final latent and PNG, five repeats. A change that reorders a sum is not exact: it is a
  separate, labelled candidate that passes the float64 kernel contract with no worse worst-column error.
- **Lossy:** against the default picture at the same seed, LPIPS median at most 0.08 and worst at most 0.15 (card
  40's cascade bar, so lossy cards compare and stack), PSNR median at least 30 dB, no newly wrong rendered text by
  OCR, object count or spatial relation in the corpus, and teacher-forced velocity cosine at least 0.999 at all six
  heads.

## Proposals, ranked

Ranked by expected gain times confidence. "Sweeps" counts how many of the three proposed it independently.

| Rank | Short name | Sweeps | Kind | Transformer | End to end, 2048 / 4096 | Cost |
|---|---|---|---|---|---|---|
| 1 | `coarse-core` | 3 | lossy, opt-in | x1.7 planned, x2.9 ceiling | x1.33 / x1.58 planned | 1 + 8-12 days |
| 2 | `exact-kernels` | 3 | exact | x1.25-1.6 | x1.12-1.29 / x1.20-1.46 | 6-12 days |
| 3 | `warm-pipeline` | 1 | exact | none | x1.3 / x1.08 | 5-7 days |
| 4 | `step-reuse` | 3 | lossy, opt-in | x1.2-1.6 | x1.08-1.29 / x1.16-1.46 | 0.5 + 8-14 days |
| 5 | `far-field-keys` | 1 | lossy, opt-in | x1.38 at 4096 | x1.05 / x1.30 | 5-7 days |
| 6 | `pv-gate` | 1 | bounded, opt-in | x1.10-1.18 at 4096 | x1.03 / x1.08-1.15 | 1 + 3-4 days |
| 7 | `m5-prompt` | 1 | exact | none | x1.05-1.08 / x1.01-1.02 | 5-9 days |

The lossy rows overlap: `coarse-core` and `step-reuse` both remove middle-block work and do not multiply. The exact
rows stack on everything. The architecture sweep's model of the strongest stack, `coarse-core` (ten outer blocks,
every step) with `warm-pipeline` and `far-field-keys`, is about 30 s at 4096 on this Mac, about 3x today's model.

### 1. `coarse-core`: the middle blocks on a half-side lattice, inside each step

**Agreement: all three sweeps**, as depth-wise coarsening (architecture), residual token merging (history) and
within-step token bottlenecking (outside).

- **Mechanism.** After block *a*, average each 2 x 2 group of picture rows and keep the full-size stream. Blocks
  *a* to *b* run on the text rows and a quarter of the picture rows; each coarse row's change is then added back to
  its four fine rows, and the last blocks run at full size. A full block costs 3.71 TFLOP at 2048 and 20.4 at 4096;
  a coarse one 0.87 and 3.70. Ten outer and forty coarse blocks make a step x2.57 / x2.90 cheaper; the planning
  configuration (ten outer blocks, steps 1-4 coarse, steps 5-6 full) is x1.69 / x1.78 on the transformer.
- **Beyond what exists.** Card 40's cascade lowers the resolution for whole steps and needs a noise lift; here the
  latent stays full size, nothing is lifted, and it cuts the linear layers (62-87% of a step), which routed
  attention cannot. It stacks on the cascade's full-size steps. Token merging libraries average tokens into
  clusters; this adds a coarse change back to a kept fine stream, on the model's own half lattice: at 4096 the
  coarse rows sit on a native 2048 canvas's lattice shifted by a constant, so their rotary phases are in range.
- **Expected gain.** Transformer x1.7 planned, x2.9 ceiling; end to end x1.33 / x1.58 planned, x1.59 / x2.23 ceiling.
  2048 is the riskier size: its coarse lattice is a 1024 canvas's, seen in training only as references.
- **Cost.** A one-day diagnostic, then 8-12 days. Two small kernels (pool, scatter-add) and a second row count and
  tile map. No new weights; 44 / 176 MB more memory.
- **A/B.** The shared protocol, sweeping *a*, *b* and the coarse steps on the calibration prompts first.
- **Speed gate.** Transformer at most card 34's six-step time / 1.5 at 4096, and the lower 95% bound of the paired
  end-to-end speed-up at least x1.5 at 4096.
- **Quality guard.** The full lossy guard. Off: byte-identical. Both kernels in the float64 contract with planted
  controls.
- **Risks.** The model never saw a resolution change in depth: blur or a 2 x 2 grid at the late steps. Merged rows
  shift the stream's statistics and may need an RMS-preserving merge.

### 2. `exact-kernels`: shape-specific products and the SwiGLU store

**Agreement: all three sweeps** (shape-dispatched GEMMs in history and outside; the SwiGLU epilogue in
architecture and outside).

- **Mechanism.** Today one 64 x 128 x 256 geometry serves QKV, the attention output and both MLP products. Measure
  a small set of tile and traversal schedules per projection class at its real row counts and dispatch the best;
  fold the SwiGLU into the first MLP product's store, so the `wide` buffer (0.23 / 0.94 GB) is never written: 23.8 of
  about 73 GB of a 2048 step's row-kernel traffic. Add a dense attention specialisation that drops the routed
  kernel's tile lists and row maps from the dense path.
- **Beyond what exists.** The tensor units are already in use; this is measured, H3-specific scheduling and one
  fusion nobody ships for it. No precision change.
- **Expected gain.** Projections x1.3-1.8, attention x1.5-2.5 on the dense path; transformer x1.25-1.6; end to end
  x1.12-1.29 / x1.20-1.46. A component of the stack, not a 1.5x card alone.
- **Cost.** 6-12 days; no weights; no new memory.
- **A/B.** Production-shape products and attention replayed from captured inputs first, then the shared protocol.
- **Speed gate.** Production-shape replay shows at least x1.3 on the projection sum before full runs; then the six
  steps take at most card 34's six-step time / 1.3 at both sizes.
- **Quality guard.** Exact.
- **Risks.** Past rounds found tile tables that did not transfer between models and fusions that were exact but
  slower; this card measures headroom first and stops if replay shows under x1.2.

### 3. `warm-pipeline`: one resident process, overlapped loads, kept tables

**One sweep (architecture).**

- **Mechanism.** One process and one Metal device hold the text encoder, transformer and decoder (about 90 GB
  resident). Metal libraries built at compile time instead of three source compiles per picture; AdaLN tables
  stored once per checkpoint and schedule, so the 26 GB `adaln_proj` read never repeats; CPU threads page in and wire
  the 38.5 GB of step weights while the GPU runs the text encoder; the PNG compressed in parallel bands.
- **Beyond what exists.** Others keep models resident and precompute AdaLN. What is new is overlapping the step
  weights' page-in with the encoder, removing the 26 GB read from a cold start, and attacking the 39% of a 2048
  picture that is not transformer steps.
- **Expected gain.** Non-step time 9.0 to about 3.0 s at 2048 and 14.5 to about 8 s at 4096: x1.3 / x1.08.
- **Cost.** 5-7 days; no weights.
- **A/B.** The five-tool chain against the one process, cold and warm page cache.
- **Speed gate.** Card 34's load, prompt and glue times summed, divided by 3 at 2048.
- **Quality guard.** Exact.
- **Risks.** Rests on card 34's load times, the least certain row of the cost model. Wiring 90 GB contends with
  other work on the machine.

### 4. `step-reuse`: reuse middle-block work across adjacent steps

**Agreement: all three sweeps**, with low confidence in each: a shared body pass for steps 1 and 2 (architecture),
gated reuse of middle-block residual branches (history), a learned per-token reuse budget (outside).

- **Mechanism.** Step 2's input is step 1's plus 0.032 x v1, and heads 1 and 2 differ by 2.3%. Cache the middle
  blocks' ungated attention and MLP branches; on chosen steps reuse them under the current step's AdaLN gates and
  head, recomputing the outer blocks. Steps 1-2 sharing one pass gives x1.20; reusing 40 blocks on two or three steps
  gives x1.31 / x1.59.
- **Beyond what exists.** Feature caches (TeaCache, FORA, TaylorSeer) are tuned for 20-50 step schedules and their
  own reports show weak gains at one to four steps. This is head-aware: every step still applies its own head and
  gates. It is not the rejected exact reuse of reference rows.
- **Expected gain.** Transformer x1.2-1.6; end to end x1.1-1.5 / x1.15-1.55.
- **Cost.** A half-day diagnostic (row cosine of hidden states between steps, per block), then 8-14 days; no weights;
  3.3 / 13.1 GB of branch cache.
- **A/B.** The shared protocol, with the reused steps and blocks chosen on calibration prompts.
- **Speed gate.** Card 34's six-step time / 1.3 at 4096, counting refreshes and cache traffic.
- **Quality guard.** Lossy, plus teacher-forced checks at all six heads.
- **Risks.** AdaLN lines move 3-18% between steps 1 and 2; distilled steps may share little. Overlaps `coarse-core`.

### 5. `far-field-keys`: exact near attention, pooled far keys

**One sweep (architecture).**

- **Mechanism.** Picture rows ordered in 8 x 8 spatial tiles; each query tile attends its 3 x 3 neighbourhood and
  the prefix exactly, and every farther key as a 2 x 2 mean of rotated K and of V with ln 4 added to its score. No
  key is dropped, no top-k is chosen, the list is fixed per canvas. At 4096 key tiles per query fall from 257 to
  about 72: attention x3.58, a step x1.38.
- **Beyond what exists.** Card 40 drops about 80% of tiles in its last two steps; this keeps all the mass at a lower
  resolution and can run on every step.
- **Expected gain.** End to end x1.05 / x1.30. Small at 2048.
- **Cost.** 5-7 days: a pooling kernel, a per-query list, the bias through the existing mask path; 118 MB at 4096.
- **Speed gate.** Card 34's 4096 step time / 1.3. **Quality guard.** Lossy; per-block attention output cosine
  against dense on teacher-forced inputs.
- **Risks.** A mean key underestimates four spread scores, so sharp long-range heads lose mass.

### 6. `pv-gate`: skip the products of key tiles that cannot matter

**One sweep (architecture).** A key tile whose best score is over 15.25 nats below the running maximum adds under
2^-22 a key. Visit near tiles first and skip such a tile's `exp` and P x V. If 50-80% of tile pairs clear the bound,
a 4096 step is x1.10-1.18 faster. Cost: a one-day histogram (go at 50% or more at 4096), then 3-4 days. Gate: card
34's 4096 attention time / 1.3. Guard: final latent cosine at least 0.9999 and PSNR at least 45 dB, with a planted
control at a looser bound. Below the round's bar alone; a cheap add-on to `exact-kernels`.

### 7. `m5-prompt`: the text encoder on the tensor units

**One sweep (history).** The prompt path's GEMM is the portable 64 x 64 kernel and its causal attention scalar.
An M5 path is x2-4 on the encoder's own time but x1.05-1.08 / x1.01-1.02 end to end. Gate: card 34's prompt time / 2;
guard: exact or float64 contract. Below the round's bar; worth folding into `warm-pipeline` only if card 34 shows the
encoder over a fifth of a 2048 picture.

## Turned down

| Idea | Sweep | Reason |
|---|---|---|
| Resolution cascade, routed late attention, 512-pixel decoder tiles | all | Card 40's design |
| int8 / fp8 / fp4 weights or activations | all | h3-image-mlx and ComfyUI ship them; no per-core gain on the M5 here; lossy formats refused upstream |
| Plain TeaCache, FORA, TaylorSeer thresholds | outside | Tuned for long schedules; their own reports show weak gains at one to four steps |
| SageAttention | outside | NVIDIA FP8 instructions; even removing attention is only x1.12 / x1.52 on the transformer |
| "Add flash attention" | outside | The kernel is already tiled with an online softmax |
| Sliding-tile, radial or routed attention alone | outside | Overlaps card 40 and leaves the linear layers, most of a step |
| Fewer Euler steps or a two-step re-distillation | all | Needs new weights; the six heads were distilled together; the re-distillation exists |
| A shallower student backbone | outside | New weights and training compute; published pruning reaches 20%, not the 45% needed |
| Region-adaptive sampling | outside | Overlaps `coarse-core` and `step-reuse`; shown on 28-30 step schedules, not six distinct heads |
| Exact text K/V or reference-row reuse across steps | all | Attention is dense both ways, so those rows change every step |
| Freezing text rows after early blocks | architecture | Text is 1.5% / 0.4% of linear work |
| Low-rank weights | architecture | Measured near full rank |
| Batching several pictures | architecture, history | Weights are already reused over every row tile; does not cut one picture's time |
| Overlapping the decoder with the steps | architecture | Same GPU; the steps are compute-bound |
| Mixed-density canvas | architecture | No in-distribution lattice; seams |
| Folding AdaLN scale into weights, Q-K symmetry, SwiGLU sparsity | architecture | No FLOPs saved, different projections, dense products cannot skip zeros |
| Graph capture, launch fusion | history | CUDA mechanisms; one command buffer a step already, and past launch-gap ceilings were about 1% |
| SSD block streaming | all | A capacity technique, slower at 256 GB; card 40 turned it down |
| Drafting (MTP-style) for the transformer | history | Velocities are continuous; nothing to verify token for token |
| Multi-GPU sequence or pipeline parallel | outside | Throughput across devices, not one picture on one Mac |
| A tiny preview decoder or less tile overlap | outside | Removing the whole 8.37 s decode is x1.02 end to end |
| Faster prompt caching, separate processes | all | Already shipped by others; a fresh prompt cannot assume a hit |
| KV precision | history | No persistent KV cache; attention operands are rebuilt each block |

## Past results that bind

Our own speed rounds on other models set these limits. Each needs the named new evidence before it returns.

| Past result | Binds | New evidence needed |
|---|---|---|
| Launch-gap and PDL work measured about 1% of a span, within noise | Graph capture, launch fusion | A Metal trace with a larger idle share |
| GEMM tile tables tuned on one model did not help another; big gains at small rows shrank at large rows | `exact-kernels` | Production-shape replay on H3-Image's own rows |
| Exact fusions of epilogues, norms and requantisation were slower or inconclusive | `exact-kernels`' SwiGLU store | A profile where the removed traffic outweighs register pressure |
| Prefetch and wider loads were exact but not reliably faster; prefetch competed for bandwidth | `warm-pipeline`'s overlap | Idle bandwidth while the encoder runs |
| Query-tiled and clustered attention were 2-5x slower once preparation was counted | `far-field-keys`, `pv-gate` | Kernel time including list building, with no CPU routing loop |
| FP8 weights did not speed a compute-bound model; smaller quantised models started slower | int8 / fp8 rows | Faster executed arithmetic, cold start included |
| Drafting that won in replay lost in served runs; fewer rounds did not mean faster runs | `step-reuse` | Measured end-to-end time, refreshes included |
| Short windows and fixed locality policies broke distant-copy cases | `coarse-core`, `far-field-keys` | Repeated textures and symmetric objects in the corpus |
| Whole-model gates failed where every block passed (an earlier image model) | every lossy card | Trajectory gates on six heads, not block checks |
| A native port was slower than an MLX runner on one image model | `exact-kernels` | The same-machine h3-image-mlx rerun |
| A candidate that passed quality failed its latency gate | every card | Both gates pass independently |

## Ready cards

Aimed at a large gain: cards 1-3 together target at least 2x end to end at 4096 against card 34 on this Mac (the
cost model gives about 2.3x at their planned gains). Order is the order to build; cards 1, 4 and 5 start with a
diagnostic and close as a measured negative if it fails.

**1. MiniMax-H3-Image: the middle blocks on a half-side lattice inside each step** (`coarse-core`)

- A one-day diagnostic first: per block and step, on ten calibration prompts, the fraction of the picture rows'
  squared norm that lies in each row's difference from its 2 x 2 group mean; go only if under 10% for the middle
  blocks, else record the numbers and close.
- `tf-h3-image coarse=A:B:STEPS`: pool and scatter-add kernels in the float64 kernel contract with planted controls;
  without `coarse`, the final latent is byte-identical.
- On the shared corpus at 4096 on this M5 Ultra: the lower 95% bound of the paired end-to-end speed-up against card
  34 at least 1.5x (the transformer alone at 1.5x gives only about 1.39x end to end); a table per A, B, STEPS and
  size in the recipe.
- The complete lossy guard, every check required: LPIPS median at most 0.08 and worst at most 0.15, PSNR median at
  least 30 dB, no newly wrong OCR text, object count or spatial relation, and teacher-forced velocity cosine at least
  0.999 at all six heads. A setting that passes LPIPS and OCR but fails any other check is rejected.

**2. MiniMax-H3-Image: exact shape-specific products, the SwiGLU store and a dense attention path** (`exact-kernels`)

- Production-shape replay of the four projection classes and dense attention, before and after; stop with a
  recorded negative if the projection sum gains under 1.2x.
- Final latent and PNG byte-identical; a sum-reordering variant only as a separate labelled option that passes the
  float64 contract with no worse worst-column error.
- The six steps take at most card 34's six-step time / 1.3 at 2048 and 4096.

**3. MiniMax-H3-Image: one resident process with overlapped loads** (`warm-pipeline`)

- One process runs prompt, transformer and decoder; Metal libraries prebuilt; AdaLN tables kept per checkpoint and
  schedule; step weights paged in during the text encoder; PNG compressed in parallel.
- PNG byte-identical to the five-tool chain; cold and warm times recorded.
- Non-step time at most a third of card 34's at 2048.
- h3-image-mlx rerun on this M5 Ultra with its teapot prompt, seed 42, at 4096; with cards 1-3 together, end to end
  at least 2x faster than card 34's at 4096 and at least 3x faster than that rerun.

**4. MiniMax-H3-Image: head-aware reuse of middle-block work across steps** (`step-reuse`)

- A half-day diagnostic first: hidden-state row cosine between adjacent steps per block; go at 0.99 or better in
  the middle blocks, else record and close.
- Off: byte-identical. On: teacher-forced velocity cosine at least 0.999 at all six heads, the lossy guard, and the
  six-step time at least 1.3x faster than card 34's at 4096 with refreshes counted.

**5. MiniMax-H3-Image: pooled far-field keys** (`far-field-keys`)

- A one-day diagnostic first: on teacher-forced inputs at 4096, per block, the attention output cosine of pooled
  far keys against dense; go at 0.999 or better in every block, else record and close.
- Pooling kernel and fixed per-query lists in the float64 contract; off: byte-identical.
- At 4096: step at least 1.3x faster than card 34's; per-block attention cosine against dense recorded; the lossy
  guard; an A/B against card 40's routed attention at equal step time.

## What round 2 starts from

Round 2 reads these measurements, not this report's estimates:

- **Card 34:** every stage's time at 2048 and 4096, cold and warm, and peak footprint. It replaces the cost model
  above; any proposal whose stage turns out under a tenth of the total drops.
- **This round's diagnostics:** the 2 x 2 energy share per block (card 1), production-shape headroom per projection
  class (card 2), adjacent-step hidden cosine per block (card 4), and the far-tile score histogram (`pv-gate`). Each
  is a result whether it says go or stop.
- **Each build card's A/B table:** speed and LPIPS per setting. A setting that fails its guard enters the binding
  table above with its numbers; one that passes sets the new baseline for the next card's gate.
- **Card 40's cascade and routed attention results**, so round 2 knows which lossy cards stack and which overlap.
- **The same-machine h3-image-mlx run**, which replaces the cross-machine 356.84 s as the external gate.

Round 2 repeats the three sweeps from those numbers, with the binding table extended by round 1's own negatives.
