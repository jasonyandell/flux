"""Hex-grid construction for v2.

Generates an N-cell hexagon-shaped board of the requested radius, with each
cell's six direct-neighbor slots resolved to cell ids (or -1 if the slot
points off the grid).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .state import (
    DEAD,
    HEX_DIRECTIONS,
    K,
    NEUTRAL,
    State,
)

SQRT3 = math.sqrt(3.0)


def axial_to_pixel(q: int, r: int) -> tuple[float, float]:
    return (SQRT3 * q + (SQRT3 / 2.0) * r, (3.0 / 2.0) * r)


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    dq, dr = q1 - q2, r1 - r2
    return max(abs(dq), abs(dr), abs(dq + dr))


def make_board(
    radius: int,
    num_players: int,
    seat_cells: Optional[np.ndarray] = None,
    dead_cells: Optional[np.ndarray] = None,
    seat_strength: float = 30.0,
    rng: Optional[np.random.Generator] = None,
) -> State:
    """Build the initial v2 board.

    - `seat_cells`: (num_players,) int — cell ids for each seat. None →
      deterministic evenly-spaced perimeter (matches v1's default).
    - `dead_cells`: (num_dead,) int — cell ids marked DEAD. None → no dead.
    """
    cells: list[tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            cells.append((q, r))
    N = len(cells)
    id_by_coord = {c: i for i, c in enumerate(cells)}

    pos = np.zeros((N, 2), dtype=np.float32)
    coord = np.zeros((N, 2), dtype=np.int32)
    for i, (q, r) in enumerate(cells):
        x, y = axial_to_pixel(q, r)
        pos[i] = (x, y)
        coord[i] = (q, r)

    neighbors = np.full((N, K), -1, dtype=np.int32)
    for i, (q, r) in enumerate(cells):
        for k, (dq, dr) in enumerate(HEX_DIRECTIONS):
            j = id_by_coord.get((q + dq, r + dr))
            if j is not None:
                neighbors[i, k] = j

    owner = np.full(N, NEUTRAL, dtype=np.int32)
    strength = np.zeros(N, dtype=np.float32)
    # Default starting strength for empty cells.
    strength[:] = 10.0

    # Determine seats.
    if seat_cells is None:
        # Deterministic perimeter sort by atan2 — mirrors v1's layout.
        perimeter = [
            i for i, (q, r) in enumerate(cells)
            if max(abs(q), abs(r), abs(q + r)) == radius
        ]
        perimeter.sort(key=lambda i: math.atan2(float(pos[i, 1]), float(pos[i, 0])))
        seat_cells = np.array(
            [perimeter[(p * len(perimeter)) // num_players] for p in range(num_players)],
            dtype=np.int32,
        )

    if dead_cells is not None and len(dead_cells) > 0:
        owner[dead_cells] = DEAD
        strength[dead_cells] = 0.0

    for p, cell in enumerate(seat_cells[:num_players]):
        c = int(cell)
        if owner[c] == DEAD:
            raise ValueError(f"seat {p} placed on a dead cell ({c})")
        owner[c] = p
        strength[c] = seat_strength

    outflow = np.zeros((N, K), dtype=np.bool_)
    edge_pressure = np.zeros((N, K), dtype=np.float32)

    return State(
        N=N,
        pos=pos,
        coord=coord,
        neighbors=neighbors,
        owner=owner,
        strength=strength,
        outflow=outflow,
        edge_pressure=edge_pressure,
        tick=0,
        num_players=num_players,
    )


def _live_subgraph_connected(is_dead: np.ndarray, neighbors: np.ndarray) -> bool:
    """BFS over live cells: True iff every live cell is reachable from any one
    live start cell. Used as a guard when sampling dead cells, so we never
    produce a board with isolated live regions.
    """
    N = is_dead.shape[0]
    K_local = neighbors.shape[1]
    # Pick any live start.
    start = -1
    for c in range(N):
        if not is_dead[c]:
            start = c
            break
    if start < 0:
        return True                          # no live cells: vacuously connected
    visited = np.zeros(N, dtype=np.bool_)
    visited[start] = True
    stack = [start]
    while stack:
        c = stack.pop()
        for k in range(K_local):
            d = int(neighbors[c, k])
            if d >= 0 and not is_dead[d] and not visited[d]:
                visited[d] = True
                stack.append(d)
    total_live = int((~is_dead).sum())
    return int(visited.sum()) == total_live


def random_seat_and_dead(
    N: int,
    num_players: int,
    num_dead_cells: int,
    rng: np.random.Generator,
    neighbors: Optional[np.ndarray] = None,
    min_seat_dist: int = 2,
    coord: Optional[np.ndarray] = None,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample dead cells (preserving live-subgraph connectivity when
    `neighbors` is given) plus seat cells with a min-distance constraint.

    Connectivity guard: when `neighbors` is provided, each candidate dead
    cell is only accepted if the remaining live cells still form a single
    connected component. If we run out of safe candidates before reaching
    `num_dead_cells`, returns however many we did manage to place.

    Seat placement: falls back to smaller min_seat_dist if necessary.
    """
    # ---- dead-cell sampling ----
    if num_dead_cells <= 0:
        dead = np.zeros(0, dtype=np.int32)
        is_dead = np.zeros(N, dtype=np.bool_)
    elif neighbors is None:
        # Old behavior: independent uniform sample.
        dead = rng.choice(N, size=num_dead_cells, replace=False).astype(np.int32)
        is_dead = np.zeros(N, dtype=np.bool_)
        is_dead[dead] = True
    else:
        is_dead = np.zeros(N, dtype=np.bool_)
        placed = 0
        candidates = np.arange(N)
        rng.shuffle(candidates)
        for c in candidates:
            if placed >= num_dead_cells:
                break
            c = int(c)
            if is_dead[c]:
                continue
            is_dead[c] = True
            if _live_subgraph_connected(is_dead, neighbors):
                placed += 1
            else:
                is_dead[c] = False
        dead = np.where(is_dead)[0].astype(np.int32)

    avail = np.where(~is_dead)[0]
    if avail.shape[0] < num_players:
        raise RuntimeError(
            f"not enough live cells for {num_players} seats "
            f"({avail.shape[0]} live, {len(dead)} dead)"
        )

    # ---- seat placement ----
    for dist in (min_seat_dist, max(1, min_seat_dist - 1), 1, 0):
        for _ in range(max_attempts):
            seats = rng.choice(avail, size=num_players, replace=False)
            if dist == 0 or coord is None:
                return seats.astype(np.int32), dead
            ok = True
            for i in range(num_players):
                if not ok:
                    break
                qi, ri = int(coord[seats[i], 0]), int(coord[seats[i], 1])
                for j in range(i + 1, num_players):
                    qj, rj = int(coord[seats[j], 0]), int(coord[seats[j], 1])
                    if hex_distance(qi, ri, qj, rj) < dist:
                        ok = False
                        break
            if ok:
                return seats.astype(np.int32), dead
    raise RuntimeError("random seat sampling failed")
