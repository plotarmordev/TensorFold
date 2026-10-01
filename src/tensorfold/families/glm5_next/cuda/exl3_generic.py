"""GLM routed experts through the generic mixed-width EXL3 path.

Serves any bit width (k2 2..16) with bit-identical epilogue arithmetic
(ACT_BF16, GLM tile configs). The caller owns the generic Scratch (one per
engine buffer set, sized to its allocated rows) and passes the live row count.

The shared expert (pick == E) is excluded by the generic kernels, as in
exl3_mm; the caller combines its slot from the BF16 shared MLP.
"""
from __future__ import annotations

import torch

from tensorfold.cuda.exl3 import experts as generic


def scratch(ex, rows: int, slots: int, device: torch.device) -> generic.Scratch:
    """The generic kernels' buffers for up to ``rows`` rows of ``slots`` slots."""

    return generic.Scratch(ex, rows, slots, device=str(device))


def routed(x, pick, ex, s, R, limit):
    """ey[pair] fp32 for the R live rows' routed pairs; the shared slot's row is stale (caller overwrites)."""

    return generic.routed(x, pick, None, ex, s, None, R, limit, act_mode=generic.ACT_BF16)
