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
| attention inputs: AR, gates (gated visits: gates, zeta, blend), norms | 2 / 5 | 2 / 5 |
| projections | 1 | 1 |
| `attendSegments` | 5 | **4** |
| wo, attention partial, residual | 3 | 3 |
| MLP gates (gated visits), norm | 1 / 4 | 1 / 4 |
| router (scores, top-8) | 2 | 2 |
| routed experts (group, gu, down, combine) and MLP partial | 5 | 5 |
| **a visit, ungated / gated** | **19 / 25** | **18 / 24** |
| **a forward** (24 ungated + 24 gated visits, 2 resets, embed and tail) | **1,067** | **1,019** |

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
48 fewer a forward (−4.5%). Every head still runs the same instructions on the same inputs, so the outputs are the
same bits. Only the grouping of launches between barriers changes.

Predicted (D): 48 × 10 µs ≈ 0.49 ms of a 27.3 ms round, about +1.8% tok/s at one stream.

## Exactness and speed

Host: `zig build test test-golden`, `test-inkling2-metal-host` (the `PrepArgs` ABI at 40 bytes), and an offline
`xcrun metal` compile of `attention.metal` all pass. The device gates (DSpark drafts equal no drafts, the
sixteen-stream chain, four-stream leave and join, greedy against the float64 reference) and the speed window are
`run.sh`'s. It needs a Mac GPU window (about 7 minutes of device tests, a 21-minute float64 gate that is mostly CPU,
two ReleaseFast builds, then 3 × 3 three-rep bench sessions):

```
docs/receipts/inkling-2-mini-metal-262/run.sh CONTROL_TREE   # CONTROL_TREE: a checkout of 1cb6ac2a9
```
