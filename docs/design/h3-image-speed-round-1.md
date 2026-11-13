# MiniMax-H3-Image speed, round 1: three independent sweeps, ranked

This round looked for ways to make one MiniMax-H3-Image picture at least 1.5x faster end to end on an M5 Ultra
(256 GB) running TensorFold's native Metal port. Three sweeps worked apart, each on a different model family, and
none saw another's proposals:

| Sweep | Question | Model family |
|---|---|---|
| Architecture | Where one picture's time goes in the text encoder, transformer and decoder, and what the maths allows | Claude Fable 5.2 |
| History | What our own past speed rounds (GLM, Nemotron, DeepSeek, Drama, Ornith, the H3 video DiT) teach an image model | GPT-6 Astra |
| Outside | Papers and other runners, as raw material, for what nobody has combined or run on a Mac | GLM-5.3 |

The outside sweep ran in a checkout of our main that does not contain this report, with read-only tools and web
search. An earlier outside pass shared the history sweep's model family; it is discarded and nothing below rests on
it. Agreement between sweeps is the confidence signal; a lone idea needs its diagnostic first. Nothing below was run
on a GPU for this round.

## What this was made on

- **TensorFold 1.2.7**, our main at `d801270c0`, merged into this branch.
- **The H3-Image implementation analysed** is the maintainer's unmerged pull requests, at these heads: #2766
  transformer `9692b83d5331`, #2779 prompt path `0fa460a1c602`, #2780 decoder `9eb514c504ca`, #2781 end to end
  `fa0e45c77b68`, #2775 VAE host rules `f5dad8bcac70`, #2778 edit encoder `dd4ec5e3c017`, #2796 prompt cache
  `e782e6f20f30`, #2808 kernel contract `8e31db228219`, #2810 edits `32e520d08f58`.
- **Measured, inherited:** card 34's stage times, card 35's cold read-ahead runs and cards 50 and 51's resident
  server, all on this M5 Ultra and recorded in [the Metal recipe](../recipes/minimax-h3-image-metal.md);
  h3-image-mlx's 356.84 s on an M5 Pro. **Estimated:** every gain and cost in this report.

## Where a picture's time goes

Measured on this M5 Ultra, card 34's prompt, seed 42, seconds:

| Path | 2048 x 2048 | 4096 x 4096 | Source |
|---|---|---|---|
| One step, GPU | 1.83 | 9.80-9.86 | card 34 |
| Six steps | 11.1 | 59.4 | card 34 |
| One-shot tools, each mapping its weights from disk | 71.4 | 189.4 | card 34 |
| Cold, every weight file evicted, read ahead | 61 (prompt 29, transformer 28 of which load 12.7, decoder 4) | 123 (13, 78 of which 16.9, 32) | card 35 |
| Warm, one resident process | 12.70-12.77 (prompt 0.22, steps 11.1, decoder and PNG 1.4) | about 100, single run | cards 51, 35 |

So the steps are 87% of a warm 2048 picture and about 60% of a warm 4096 one. At 4096 the rest is mostly the decoder
stage's host tail: its 441 tiles take 4.2 s of GPU, and single-threaded PNG deflate takes most of the remainder.

The architecture sweep's split of a step, at about 100 TFLOPS of bf16 on the tensor units:

| Part of a step | 2048 | 4096 | Bound by |
|---|---|---|---|
| Linear layers | 87% | 62% | compute, cache-served |
| Attention | 13% | 38% | compute and `exp` |

Facts the proposals rest on, checked in the port's code:

- **Attention is dense both ways.** Text rows query picture keys and take each step's timestep line, so no text K/V
  is constant across steps or blocks. Any cross-step reuse is approximate.
- **The linear layers are not DRAM-bound.** One 64 x 128 x 256 GEMM geometry serves all four projection classes;
  the SwiGLU's `wide` buffer is written and read back once a block.
- **The weights are near full rank** (block 25 needs rank 4,343-5,189 of 5,376 for 99% of the energy).
- **The six step heads are small linear maps** (96 x 5,376 and a bias) and differ by only 1.5-2.3% from each other.
- **The picture lattice is oversampled in position.** At 4096 the largest rotary phase between neighbouring rows is
  0.25 rad, and a 2 x 2 mean keeps at least 99.2% of the top frequency's magnitude.

## Guardrails every proposal meets

- **Not already someone's.** Card 40's design (commit `bcc2f15c`: the resolution cascade, routed attention in the
  last steps, the decoder in 512-pixel tensor-unit tiles), [main's ideas page](../recipes/minimax-h3-image-ideas.md)
  (the cascade, pooled far-field keys `far=pool`, the GPU and SME2 row split), cards 35, 50 and 51 (read-ahead, the
  resident server, resident weights), the maintainer's open H3-Image pull requests (#2766, #2775, #2778-#2781,
  #2796, #2808, #2810) and h3-image-mlx (int8 W8A8, separate processes, resident or streamed weights, precomputed
  AdaLN, batched decoder tiles) count only where a proposal clearly goes beyond them, said in its row. So do the CUDA
  runners' graphs, compile, fused QK-norm and rotary, conditioning caches and FP8 repacks.
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
- **Runs:** the same binary with the change off and on, alternated A-B-B-A in one resident process with no other GPU
  work; cold runs, every weight file evicted, reported apart from warm ones.
- **Recorded:** wall time to the PNG, each stage's time, each step's GPU time, peak physical footprint.

### Speed gates

- **Local:** against the measured times above: card 34's step times, card 51's warm 2048 picture (12.70-12.77 s) and
  card 35's cold pictures (61 s and 123 s). A speed-up claim needs the lower 95% bound of the paired speed-up at or
  above the card's figure.
- **External:** h3-image-mlx's teapot prompt, seed 42, at 4096 x 4096: 356.84 s on an M5 Pro (343.6 s of steps,
  8.37 s of decode). Our cold 4096 picture is already 123 s, but on a different machine, so it is not an algorithmic
  comparison: card 4 below reruns h3-image-mlx on this M5 Ultra.

### Quality guards

- **Exact:** byte-identical final latent and PNG, five repeats. A change that reorders a sum is not exact: it is a
  separate, labelled candidate that passes the float64 kernel contract with no worse worst-column error.
- **Lossy:** against the default picture at the same seed, LPIPS median at most 0.08 and worst at most 0.15 (card
  40's cascade bar, so lossy cards compare and stack), PSNR median at least 30 dB, no newly wrong rendered text by
  OCR, object count or spatial relation in the corpus, and teacher-forced velocity cosine at least 0.999 at every
  head that runs.

## Proposals, ranked

Ranked by expected gain times confidence. "Sweeps" counts how many of the three proposed it independently; a
dissent is a sweep that turned the same idea down. End-to-end gains are on the warm, resident picture.

| Rank | Short name | Sweeps | Kind | Transformer | Warm end to end, 2048 / 4096 | Cost |
|---|---|---|---|---|---|---|
| 1 | `fewer-calls` | 2 | lossy, opt-in | x1.2-2.0 | x1.17-1.78 / x1.11-1.42 | 0.5 + 2-3 days |
| 2 | `coarse-core` | 2, one dissent | lossy, opt-in | x1.7 planned, x2.9 ceiling | x1.55 / x1.35 planned | 1 + 8-12 days |
| 3 | `exact-kernels` | 3 | exact | x1.1-1.6 | x1.09-1.49 / x1.06-1.29 | 6-12 days |
| 4 | `cold-overlap` | 2 | exact | none | cold x1.35 / x1.17, warm none | 3-5 days |
| 5 | `step-reuse` | 2, one dissent | lossy, opt-in | x1.2-1.6 | x1.17-1.49 / x1.11-1.29 | 0.5 + 8-14 days |

`fewer-calls` cuts the number of transformer calls and `coarse-core` the cost of each, so they multiply;
`step-reuse` overlaps both. The exact rows stack on everything. With `fewer-calls` at four calls, `coarse-core`
planned and `exact-kernels` at its low end, the steps fall to about a third: warm 2048 from 12.7 s to about 5.6 s
(x2.3), warm 4096 from about 100 s to about 61 s (x1.6), where the decoder stage's host tail is then two thirds of
the picture.

### 1. `fewer-calls`: fewer transformer calls with blended step heads

**Agreement: architecture and outside**, as one shared body pass for steps 1 and 2 (architecture) and merged Euler
intervals with blended heads (outside). History did not propose it.

- **Mechanism.** Merge adjacent Euler intervals: run the 50 blocks once per merged interval at its first sigma, and
  apply the weighted mean of the merged steps' heads (weights from each step's share of the sigma span, matrix and
  bias blended once on the host). `steps=6` stays the default and byte-identical; `steps=5`, `4` and `3` are opt-in.
- **Beyond what exists.** The turned-down "fewer steps" rows need new weights; this needs none. It exists only
  because this model keeps a separate small head per step, and none of the runners use that. Interval-average
  velocities (MeanFlow) are the precedent for why a wide step wants a blended, not an endpoint, velocity.
- **New evidence against the rejection.** The six heads differ by only 1.5-2.3%, so a blend stays near both.
- **Expected gain.** Calls 6 to 5, 4 or 3: transformer x1.2, x1.5 or x2.0. Warm 2048 12.7 to 10.9, 9.0 or 7.2 s
  (x1.17-1.78); warm 4096 about 100 to 90, 80 or 70 s (x1.11-1.42).
- **Cost.** A half-day diagnostic, then 2-3 days: a host schedule and head blend, no kernels, no weights.
- **A/B.** The shared protocol at each call count, stopping at the fewest calls that pass every check.
- **Speed gate.** At four calls: six-step time at most card 34's / 1.45 (7.7 s at 2048, 41 s at 4096) and the warm
  2048 picture at most 9.5 s.
- **Quality guard.** The full lossy guard against the six-step picture, at every call count claimed.
- **Risks.** The distilled heads expect their own sigma's input; a wide step may lose fine detail and small text.
  The diagnostic tells early.

### 2. `coarse-core`: the middle blocks on a half-side lattice, inside each step

**Agreement: architecture and history**, as depth-wise coarsening (architecture) and residual token merging
(history). **Dissent: outside**, which turned down token merging because merged rows leave the 2 x 2 patch
distribution the model learned.

- **Mechanism.** After block *a*, average each 2 x 2 group of picture rows and keep the full-size stream. Blocks
  *a* to *b* run on the text rows and a quarter of the picture rows; each coarse row's change is then added back to
  its four fine rows, and the last blocks run at full size. Ten outer and forty coarse blocks make a step x2.57 /
  x2.90 cheaper; the planning configuration (ten outer blocks, steps 1-4 coarse, steps 5-6 full) is x1.69 / x1.78
  on the transformer.
- **Beyond what exists.** The cascade lowers the resolution for whole steps and needs a lift; here the latent stays
  full size, nothing is lifted, and it cuts the linear layers, which routed or pooled attention cannot. Token merging
  averages tokens into clusters and drops them; this adds a coarse change back to a kept fine stream, which answers
  the dissent: the stream that leaves the middle blocks is full size. At 4096 the coarse rows sit on a native 2048
  canvas's lattice shifted by a constant, so their rotary phases are in range.
- **Expected gain.** Warm 2048 12.7 to 8.2 s (x1.55), warm 4096 about 100 to 74 s (x1.35), planned; x2.15 / x1.63 at
  the ceiling. 2048 is the riskier size: its coarse lattice is a 1024 canvas's.
- **Cost.** A one-day diagnostic, then 8-12 days. Two small kernels (pool, scatter-add) and a second row count and
  tile map. No new weights; 44 / 176 MB more memory.
- **A/B.** The shared protocol, sweeping *a*, *b* and the coarse steps on the calibration prompts first.
- **Speed gate.** Six-step time at most card 34's / 1.5 at both sizes (7.4 s and 39.6 s), and the lower 95% bound of
  the paired warm 2048 speed-up at least x1.4.
- **Quality guard.** The full lossy guard. Off: byte-identical. Both kernels in the float64 contract with planted
  controls.
- **Risks.** The model never saw a resolution change in depth: blur or a 2 x 2 grid at the late steps. Merged rows
  shift the stream's statistics and may need an RMS-preserving merge.

### 3. `exact-kernels`: shape-specific products and the SwiGLU store

**Agreement: all three sweeps** (shape-dispatched GEMMs in history, the SwiGLU epilogue in architecture,
concatenated QKV and gate-up products with retuned tiles in outside).

- **Mechanism.** Today one 64 x 128 x 256 geometry serves QKV, the attention output and both MLP products. Measure
  a small set of tile and traversal schedules per projection class at its real row counts and dispatch the best;
  concatenate products that share an input along N where that keeps each output's sum order; fold the SwiGLU into the
  first MLP product's store, so the `wide` buffer (0.23 / 0.94 GB) is never written. Add a dense attention
  specialisation that drops the routed kernel's tile lists and row maps from the dense path.
- **Beyond what exists.** The tensor units are already in use; this is measured, H3-specific scheduling and one
  fusion nobody ships for it. No precision change.
- **Expected gain.** Transformer x1.1-1.25 (outside) to x1.25-1.6 (architecture); warm 2048 12.7 to 11.6-8.5 s
  (x1.09-1.49), warm 4096 about 100 to 94-78 s (x1.06-1.29). It speeds the default path, so it stacks under every
  other card.
- **Cost.** 6-12 days; no weights; no new memory.
- **A/B.** Production-shape products and attention replayed from captured inputs first, then the shared protocol.
- **Speed gate.** Production-shape replay shows at least x1.2 on the projection sum before full runs; then the six
  steps at most card 34's / 1.2 (9.2 s at 2048, 49.5 s at 4096).
- **Quality guard.** Exact.
- **Risks.** Past rounds found tile tables that did not transfer between models and fusions that were exact but
  slower; this card measures headroom first and stops if replay shows under x1.2.

### 4. `cold-overlap`: the first picture's loads behind its compute

**Agreement: architecture and outside**, as Metal libraries built once instead of compiled from source per tool,
and page-in of the next stage's weights overlapped with the running one.

- **Mechanism.** Build the Metal libraries at compile time, or cache them by source hash and OS build. While the
  prompt encoder pages in and runs, page in and wire the transformer's step weights; while the steps run, page in the
  decoder's.
- **Beyond what exists.** Card 35's read-ahead warms each tool's own files at its own start; cards 50 and 51 keep
  weights resident once loaded. Nothing overlaps one stage's loading with another's work on a first picture.
- **Expected gain.** Cold 2048 61 to about 45 s (x1.35), cold 4096 123 to about 105 s (x1.17), if the disk has
  headroom while the prompt pages in. Warm pictures do not change.
- **Cost.** 3-5 days; no weights.
- **A/B.** Card 35's protocol: every weight file evicted, off and on, A-B-B-A.
- **Speed gate.** Cold 2048 at most 48 s and cold 4096 at most 110 s.
- **Quality guard.** Exact.
- **Risks.** Rests on disk headroom during the prompt stage; if read-ahead already saturates the disk, the overlap
  gains little and the card closes on its first measurement.

### 5. `step-reuse`: reuse middle-block work across adjacent steps

**Agreement: architecture and history**, with low confidence in each: gated reuse of middle-block residual branches.
**Dissent: outside**, which turned down cross-step caches because six wide sigma steps are their worst case.

- **Mechanism.** Cache the middle blocks' ungated attention and MLP branches; on chosen steps reuse them under the
  current step's AdaLN gates and head, recomputing the outer blocks.
- **Beyond what exists.** Feature caches (TeaCache, FORA, TaylorSeer) are tuned for 20-50 step schedules; this is
  head-aware, every step applies its own head and gates. It is not the rejected exact reuse of reference rows.
- **Expected gain.** Transformer x1.2-1.6; warm 2048 x1.17-1.49, warm 4096 x1.11-1.29.
- **Cost.** A half-day diagnostic (row cosine of hidden states between steps, per block), then 8-14 days; 3.3 /
  13.1 GB of branch cache.
- **Speed gate.** Six-step time at most card 34's / 1.3 at 4096, refreshes and cache traffic counted.
- **Quality guard.** Lossy, plus teacher-forced checks at all six heads.
- **Risks.** AdaLN lines move 3-18% between steps 1 and 2. It overlaps `fewer-calls` and `coarse-core`, and is not
  a card while they are open.

## Turned down

| Idea | Sweep | Reason |
|---|---|---|
| Resolution cascade, routed late attention, 512-pixel decoder tiles | all | Card 40's design |
| Pooled far-field keys | architecture | Main's ideas page already claims it as `far=pool` |
| Hierarchical far field (dense core, 2 x 2 ring, 4 x 4 remainder) | outside | About x1.09 over `far=pool` on 4096 steps, x1.04 end to end; a setting for that card to try, not a card |
| GPU and SME2 row split | none | Main's ideas page claims it; at most x1.07 |
| Resident process, kept AdaLN tables, prompt and transformer in one process | architecture | Landed as cards 50 and 51; warm 2048 is 12.7 s |
| Skipping far key tiles by a logit bound | architecture | QK-norm keeps logits in a narrow band (outside), so few tiles clear the bound; under x1.18 at its best |
| Text encoder on the tensor units | history | Measured 0.21-0.53 s of GPU a picture; the cold prompt time is page-in |
| Low-rank weight factors | outside | Measured near full rank: 99% of the energy needs rank 4,343-5,189, against the 2,688 the gain needs |
| int8 / fp8 / fp4 / MX weights or activations, int8 attention scores | all | h3-image-mlx and ComfyUI ship them; ComfyUI's int8 steps here are 1.7-1.9x slower than our bf16; lossy formats refused upstream |
| Plain TeaCache, FBCache, DeepCache, PAB thresholds; static block skipping | outside | Tuned for long schedules; six wide steps are their worst case |
| Fewer Euler steps or a two-step re-distillation | all | Needs new weights; `fewer-calls` is the training-free form |
| Exact text K/V or reference-row reuse across steps | all | Attention is dense both ways, so those rows change every step |
| Token merging or pruning that drops rows | outside, history | Every row is a picture patch; `coarse-core` keeps the fine stream instead |
| Linear or random-feature far-field attention | outside | Unbounded error where pooling's is bounded |
| Freezing text rows after early blocks | architecture | Text is 1.5% / 0.4% of linear work |
| Batching several pictures | architecture, history | Weights are already reused over every row tile; does not cut one picture's time |
| Overlapping the decoder with the steps | architecture | Same GPU; the steps are compute-bound |
| Mixed-density canvas; a 2048 canvas upscaled to 4096 | architecture, outside | No in-distribution lattice; the cascade dominates it |
| Folding AdaLN scale into weights, Q-K symmetry, SwiGLU sparsity | architecture | No FLOPs saved, different projections, dense products cannot skip zeros |
| First-level Strassen on the products | outside | About x1.12 on the steps at best, and it changes sum order |
| Local-only heads skipping far keys | outside | About x1.04 on top of pooling |
| Neural Engine co-execution | outside | fp16 conversion breaks the kernel contract; weeks of porting for an unmeasured split |
| Graph capture, launch fusion | history | CUDA mechanisms; one command buffer a step already, and past launch-gap ceilings were about 1% |
| SSD block streaming | all | A capacity technique, slower at 256 GB; card 40 turned it down |
| Drafting for the transformer | history, outside | Velocities are continuous; nothing to verify token for token |
| Multi-GPU sequence, pipeline or stale-KV parallel | outside | Saves communication across devices, not one picture's work on one GPU |
| Parallel PNG bands or another deflate | outside | Claimed in the Metal recipe, and they change PNG bytes, which the byte-identity gate forbids without the lead's approval |
| Prompt caching, separate processes | all | Already shipped by others; a fresh prompt cannot assume a hit |
| KV precision | history | No persistent KV cache; attention operands are rebuilt each block |

## Past results that bind

Our own speed rounds on other models set these limits. Each needs the named new evidence before it returns.

| Past result | Binds | New evidence needed |
|---|---|---|
| Launch-gap and PDL work measured about 1% of a span, within noise | Graph capture, launch fusion | A Metal trace with a larger idle share |
| GEMM tile tables tuned on one model did not help another; big gains at small rows shrank at large rows | `exact-kernels` | Production-shape replay on H3-Image's own rows |
| Exact fusions of epilogues, norms and requantisation were slower or inconclusive | `exact-kernels`' SwiGLU store | A profile where the removed traffic outweighs register pressure |
| Prefetch and wider loads were exact but not reliably faster; prefetch competed for bandwidth | `cold-overlap` | Idle disk bandwidth while the prompt stage pages in |
| Query-tiled and clustered attention were 2-5x slower once preparation was counted | far-field settings of `far=pool` | Kernel time including list building, with no CPU routing loop |
| FP8 weights did not speed a compute-bound model; smaller quantised models started slower | int8 / fp8 rows | Faster executed arithmetic, cold start included |
| Drafting that won in replay lost in served runs; fewer rounds did not mean faster runs | `step-reuse`, `fewer-calls` | Measured end-to-end time, refreshes included |
| Short windows and fixed locality policies broke distant-copy cases | `coarse-core` | Repeated textures and symmetric objects in the corpus |
| Whole-model gates failed where every block passed (an earlier image model) | every lossy card | Trajectory gates on the heads that run, not block checks |
| A native port was slower than an MLX runner on one image model | `exact-kernels` | The same-machine h3-image-mlx rerun |
| A candidate that passed quality failed its latency gate | every card | Both gates pass independently |

## Ready cards

Aimed at a large gain: cards 1-3 together target at least 2x on a warm 2048 picture against card 51's 12.7 s (the
planned gains give about 2.3x). Order is the order to build; cards 1 and 2 start with a diagnostic and close as a
measured negative if it fails.

**1. MiniMax-H3-Image: fewer transformer calls with blended step heads** (`fewer-calls`)

- A half-day diagnostic first: on the ten calibration prompts at 2048, the final latent's cosine against six steps
  at four calls; go at 0.98 or better, else record the numbers and close.
- `tf-h3-image steps=N` with N in 3-6 and each merged interval's head blended on the host; without it, or at
  `steps=6`, the final latent is byte-identical.
- On the shared corpus on this M5 Ultra, at the fewest calls that pass: six-step time at most card 34's / 1.45 at
  both sizes, and the warm 2048 picture at most 9.5 s in the resident server.
- The complete lossy guard against the six-step picture, every check required; a table per N and size in the recipe.

**2. MiniMax-H3-Image: the middle blocks on a half-side lattice inside each step** (`coarse-core`)

- A one-day diagnostic first: per block and step, on the ten calibration prompts, the fraction of the picture rows'
  squared norm that lies in each row's difference from its 2 x 2 group mean; go only if under 10% for the middle
  blocks, else record the numbers and close.
- `tf-h3-image coarse=A:B:STEPS`: pool and scatter-add kernels in the float64 kernel contract with planted controls;
  without `coarse`, the final latent is byte-identical.
- On the shared corpus on this M5 Ultra: six-step time at most card 34's / 1.5 at both sizes, and the lower 95% bound
  of the paired warm 2048 speed-up at least 1.4x; a table per A, B, STEPS and size in the recipe.
- The complete lossy guard, every check required, alone and stacked with card 1's chosen `steps`.

**3. MiniMax-H3-Image: exact shape-specific products, the SwiGLU store and a dense attention path** (`exact-kernels`)

- Production-shape replay of the four projection classes and dense attention, before and after; stop with a
  recorded negative if the projection sum gains under 1.2x.
- Final latent and PNG byte-identical; a sum-reordering variant only as a separate labelled option that passes the
  float64 contract with no worse worst-column error.
- The six steps take at most card 34's / 1.2 at 2048 and 4096.

**4. MiniMax-H3-Image: the first picture's loads behind its compute** (`cold-overlap`)

- Metal libraries prebuilt or cached by source hash; transformer weights paged in during the prompt stage, decoder
  weights during the steps.
- Final latent and PNG byte-identical to card 35's read-ahead build; cold and warm times recorded.
- Cold 2048 at most 48 s and cold 4096 at most 110 s on this M5 Ultra, every weight file evicted before each run.
- h3-image-mlx rerun on this M5 Ultra with its teapot prompt, seed 42, at 4096, cold and warm; our times at that
  prompt and size beside it.

## What round 2 starts from

Round 2 reads these measurements, not this report's estimates:

- **The warm 4096 stage split** from the resident server, which no card here records yet: card 35's 100 s is one
  diagnostic run. If the decoder stage's host tail stays near 40 s, it caps every card above at 4096, and round 2
  needs the lead's decision on a decoded-pixel gate for PNG writing before any sweep can work on it.
- **This round's diagnostics:** the four-call latent cosine (card 1), the 2 x 2 energy share per block (card 2),
  production-shape headroom per projection class (card 3) and the disk's idle bandwidth during the prompt stage
  (card 4). Each is a result whether it says go or stop.
- **Each build card's A/B table:** speed and every guard metric per setting. A setting that fails its guard enters
  the binding table above with its numbers; one that passes sets the new baseline for the next card's gate.
- **Card 40's cascade and main's `far=pool` results**, so round 2 knows which lossy cards stack and which overlap.
- **The same-machine h3-image-mlx run** (card 4), which replaces the cross-machine 356.84 s as the external gate.

Round 2 repeats the three sweeps from those numbers, on three model families again, with the binding table extended
by round 1's own negatives.
