"""3-hop hex vision table for the v2 model.

Independent of the game's flow graph (which stays at hex-distance ≤ 2).
This defines what the v2 NN *sees*; flows still travel only via graph edges.

The 3-hop ball on a hex grid has 6 + 12 + 18 = 36 cells. Boundary cells have
fewer real neighbors — slots are padded with -1.
"""
from __future__ import annotations

import numpy as np

from .graph import axial_to_pixel

STRIDE_V2 = 36


def axial_cells(radius: int) -> list[tuple[int, int]]:
    """Reconstruct the cell list in the same order as make_initial_state."""
    cells: list[tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            cells.append((q, r))
    return cells


def build_vision_table(radius: int, vision_distance: int = 3) -> np.ndarray:
    """Returns (N * STRIDE_V2,) int32. -1 = empty slot (cell near boundary).
    Neighbors per cell are sorted by (pos.x, pos.y) — matching the v1
    convention so the feature build maintains a consistent neighbor order.
    """
    cells = axial_cells(radius)
    id_by = {(q, r): i for i, (q, r) in enumerate(cells)}
    n = len(cells)
    out = np.full(n * STRIDE_V2, -1, dtype=np.int32)

    def _key(nid: int) -> tuple[float, float]:
        p = axial_to_pixel(*cells[nid])
        return (p.x, p.y)

    for i, (q, r) in enumerate(cells):
        nbs: list[int] = []
        for dq in range(-vision_distance, vision_distance + 1):
            for dr in range(-vision_distance, vision_distance + 1):
                if dq == 0 and dr == 0:
                    continue
                if max(abs(dq), abs(dr), abs(dq + dr)) > vision_distance:
                    continue
                j = id_by.get((q + dq, r + dr))
                if j is not None:
                    nbs.append(j)
        nbs.sort(key=_key)
        for k in range(min(len(nbs), STRIDE_V2)):
            out[i * STRIDE_V2 + k] = nbs[k]
    return out
