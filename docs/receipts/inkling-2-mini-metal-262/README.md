# Inkling 2 Mini on Metal, M5 Ultra: card 262's receipt

Baseline: card 248's head `1cb6ac2a9` and its `../inkling-2-mini-metal-248/profile.md`. A one-stream code round is
27.3 ms: verify 16.3 ms, draft 11.0 ms. The target forward runs at 3× its 5.4 ms bandwidth floor, and its time goes
to the GPU's chain of barrier-separated stages.

## Stages a forward, counted from source

The forward encodes on one concurrent compute encoder. Its only ordering points are explicit `e.barrier()` calls
(`zig/src/metal/command.zig` `barrier`: `memoryBarrierWithScope`), and none is added automatically. Count for the
verify path (`backend.zig` `runShared`, `prompt=false`, tensor units), per layer visit:

| stage chain | before | after |
|---|---|---|
| attention inputs: AR, gates + norms (gated visits) | 2 / 5 | 2 / **2** |
| projections | 1 | 1 |
| `attendSegments` | 5 | **4** |
| wo, attention partial, residual | 3 | 3 |
| MLP gate + norm | 1 / 4 | 1 / **1** |
| router (scores + top-8) | 2 | **1** |
| routed experts (group, gu, down, combine) and MLP partial | 5 | **4** (≤32 rows, tensor units); 5 otherwise |
| **a visit, ungated / gated** | **19 / 25** | **16 / 16** (≤32 rows); 17 / 17 otherwise |
| **a forward** (24 ungated + 24 gated visits, 2 resets, embed and tail) | **1,067** | **779** (≤32 rows); 827 otherwise |

The 248 profile's estimate of about 16 stages a visit (770 a token) left out the attention inputs, the gate chains,
the router's inner barrier and the two partials. At 10.9 ms of non-weight time over 1,067 stages, a stage costs
about 10 µs (D). Every barrier guards a real read-after-write, so the only way to drop one is to move work across
it or to fuse kernels.

## The change

`attendSegments` used to run convolution → prep → relative table → partials → merge, one barrier between each.
`tf_ink2_prep`'s query and relative heads read only the projections, and the relative table reads only their output.
`attention.prep` now launches a part of the heads (`PrepArgs.unit0`): the query heads run alongside the
convolutions, and the key and value heads, which read the convolutions, run alongside the relative table. This
gives convolution + query prep → key/value prep + relative table → partials → merge, one barrier fewer a visit and
48 fewer a forward from attention (−4.5%). Every head still runs the same instructions on the same inputs, so the
outputs are the same bits. Only the grouping of launches between barriers changes.

At up to 32 rows, the gate-up tensor-unit kernel builds each expert's ordered route list in threadgroup memory,
and its first tile writes the same list to the existing device buffer for the down kernel. This replaces the separate
group launch and its barrier on each visit. Larger forwards retain the old group pass. The kernel's dot, activation
and down inputs are unchanged; contract coverage compares its outputs against the float64 reference.

The router's score and top-8 launches are now one dispatch (`tf_inkling2_route`): a threadgroup a row computes
each expert's dot in the old kernel's lane order into threadgroup memory, then runs the same pick, tie-break and
NaN scan — one barrier fewer a visit. On gated visits each controller tuple is likewise one dispatch
(`tf_inkling2_loop_gate`): normalize, zeta, blend and the visit's RMS norm run as phases of a per-row
threadgroup with threadgroup barriers between phases, with every store and rounding kept bit-identical
(`s.gated`/`s.mlp_gated` still receive the FP32 blend rows). That removes three stages at each of the two call
sites on a gated visit — the gate chain's two inner barriers plus the separate norm stage.

Predicted (D): 288 × 10 µs ≈ 2.9 ms of a 27.3 ms round, about +11% tok/s at one stream — above the 5% window
threshold, awaiting Capri's go for a Mac GPU window.

## Exactness and speed

Host: `zig build test test-golden`, `test-inkling2-metal-host` (the `PrepArgs` ABI at 40 bytes), and an offline
`xcrun metal` compile of `attention.metal` all pass. The device gates (DSpark drafts equal no drafts, the
sixteen-stream chain, four-stream leave and join, greedy against the float64 reference) and the speed window are
`run.sh`'s. It needs a Mac GPU window (about 7 minutes of device tests, a 21-minute float64 gate that is mostly CPU,
two ReleaseFast builds, then 3 × 3 three-rep bench sessions):

```
docs/receipts/inkling-2-mini-metal-262/run.sh CONTROL_TREE   # CONTROL_TREE: a checkout of 1cb6ac2a9
```
