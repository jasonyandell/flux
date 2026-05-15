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


def carve_seat_connectors(
    seats: np.ndarray, dead: np.ndarray, neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bridge every live component into one connected live subgraph by
    reviving dead cells along shortest-cost paths.

    Approach: label all live-cell components, pick the component containing
    the most seats as the "main island" (tie-broken to the largest component
    if no seat sits in any component), and for every other live component
    revive the dead cells along a min-cost path back to the main island
    (0-1 BFS: cost 0 through live cells, cost 1 through dead).

    The function bridges *every* live component, not just seat-bearing ones,
    so the result satisfies the v2 board invariant: all non-dead cells are
    reachable from every other non-dead cell. Isolated live pockets — even
    those containing no seats — would otherwise sit dormant, immune to
    capture and useless to the solvers.

    Returns (new_dead_array, carved_cells). The carved cells are the
    previously-dead cells revived to bridge components. Useful when you want
    the spatial character of a uniform-random dead-cell sample but a
    guaranteed connected board — typically only a handful of cells get
    carved, much less perturbation than rejecting the entire board.
    """
    from collections import deque
    N = neighbors.shape[0]
    K_local = neighbors.shape[1]
    is_dead = np.zeros(N, dtype=np.bool_)
    if len(dead) > 0:
        is_dead[dead] = True

    # Label live-cell components and remember a representative cell per comp.
    comp = np.full(N, -1, dtype=np.int32)
    comp_repr: list[int] = []
    comp_size: list[int] = []
    next_comp = 0
    for start in range(N):
        if is_dead[start] or comp[start] >= 0:
            continue
        comp[start] = next_comp
                    size += 1
                    stack.append(d)
        comp_size.append(size)
        next_comp += 1

    if next_comp <= 1:
        return dead, np.array([], dtype=np.int32)

    # Seats per component (for tie-breaking which component is the main island).
    seats_in_comp: dict[int, list[int]] = {}
    for s in seats:
        s = int(s)
        if is_dead[s]:
            continue
        seats_in_comp.setdefault(int(comp[s]), []).append(s)

    # Main island: most seats, then largest component as the tiebreaker so a
    # seat-free uniform-random board still picks the biggest blob.
    def _rank(cid: int) -> tuple[int, int]:
        return (len(seats_in_comp.get(cid, [])), comp_size[cid])

    main_comp_id = max(range(next_comp), key=_rank)
    main_target = (
        seats_in_comp.get(main_comp_id, [comp_repr[main_comp_id]])[0]
    )
    carved: list[int] = []

    for cid in range(next_comp):
        if cid == main_comp_id:
            continue
        # Prefer a seat in this component as the source if one exists; the
        # carved path then runs seat-to-seat, which usually carves fewer
        # cells than starting from an arbitrary representative.
        src_candidates = seats_in_comp.get(cid, [comp_repr[cid]])
        src = src_candidates[0]
        # 0-1 BFS: cost 0 through live cells, cost 1 through dead.
        dist = np.full(N, 10 ** 9, dtype=np.int32)
        prev = np.full(N, -1, dtype=np.int32)
        dist[src] = 0
        dq = deque([src])
        while dq:
            v = dq.popleft()
            dv = dist[v]
            for k in range(K_local):
                u = int(neighbors[v, k])
                if u < 0:
                    continue
                w = 1 if is_dead[u] else 0
                nd = dv + w
                if nd < dist[u]:
                    dist[u] = nd
                    prev[u] = v
                    if w == 0:
                        dq.appendleft(u)
                    else:
                        dq.append(u)
        if dist[main_target] >= 10 ** 9:
            continue  # off-grid? skip; shouldn't happen on a hex grid
        node = main_target
        while node != -1 and node != src:
            if is_dead[node]:
                is_dead[node] = False
                carved.append(node)
            node = int(prev[node])

    new_dead = np.where(is_dead)[0].astype(np.int32)
    return new_dead, np.array(carved, dtype=np.int32)


def max_seat_pair_distance(
    seats: np.ndarray, dead: np.ndarray, neighbors: np.ndarray,
) -> int:
    """Return the maximum BFS graph distance (in hops through non-dead cells)
    between any two seats. Returns -1 if any seat is unreachable from another
    (disconnected). On a snaking 50%-dead board this can be 2-3× the empty
    hex diameter — large enough that pressure can't traverse it in game time
    and seats effectively play their own fight."""
    from collections import deque
    N = neighbors.shape[0]
    K_local = neighbors.shape[1]
    is_dead = np.zeros(N, dtype=np.bool_)
    if len(dead) > 0:
        is_dead[dead] = True
    worst = 0
    for s_idx in range(len(seats) - 1):
        start = int(seats[s_idx])
        if is_dead[start]:
            return -1
        dist = np.full(N, -1, dtype=np.int32)
        dist[start] = 0
        q = deque([start])
        while q:
            c = q.popleft()
            for k in range(K_local):
                d = int(neighbors[c, k])
                if d >= 0 and not is_dead[d] and dist[d] < 0:
                    dist[d] = dist[c] + 1
                    q.append(d)
        for t in seats[s_idx + 1:]:
            dt = int(dist[int(t)])
            if dt < 0:
                return -1
            if dt > worst:
                worst = dt
    return worst


def seats_mutually_reachable(
    seats: np.ndarray, dead: np.ndarray, neighbors: np.ndarray,
) -> bool:
    """Return True iff every seat cell is reachable from every other seat cell
    via BFS through non-dead cells. Strong guarantee against any seat being
    placed in an isolated live-subgraph pocket — defensive even though
    `random_seat_and_dead` is supposed to maintain live-graph connectivity."""
    N = neighbors.shape[0]
    K_local = neighbors.shape[1]
    is_dead = np.zeros(N, dtype=np.bool_)
    if len(dead) > 0:
        is_dead[dead] = True
    if len(seats) == 0:
        return True
    start = int(seats[0])
    if is_dead[start]:
        return False
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
    return bool(visited[seats].all())


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


def seats_mutually_reachable(
    seats: np.ndarray, dead: np.ndarray, neighbors: np.ndarray,
) -> bool:
    """Return True iff every seat cell is reachable from every other seat cell
    via BFS through non-dead cells."""
    N = neighbors.shape[0]
    K_local = neighbors.shape[1]
    is_dead = np.zeros(N, dtype=np.bool_)
    if len(dead) > 0:
        is_dead[dead] = True
    if len(seats) == 0:
        return True
    start = int(seats[0])
    if is_dead[start]:
        return False
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
    return bool(visited[seats].all())


def max_seat_pair_distance(
    seats: np.ndarray, dead: np.ndarray, neighbors: np.ndarray,
) -> int:
    """Return the maximum BFS graph distance (in hops through non-dead cells)
    between any pair of seats. -1 if any seat is unreachable from another."""
    from collections import deque
    N = neighbors.shape[0]
    K_local = neighbors.shape[1]
    is_dead = np.zeros(N, dtype=np.bool_)
    if len(dead) > 0:
        is_dead[dead] = True
    worst = 0
    for s_idx in range(len(seats) - 1):
        start = int(seats[s_idx])
        if is_dead[start]:
            return -1
        dist = np.full(N, -1, dtype=np.int32)
        dist[start] = 0
        q = deque([start])
        while q:
            c = q.popleft()
            for k in range(K_local):
                d = int(neighbors[c, k])
                if d >= 0 and not is_dead[d] and dist[d] < 0:
                    dist[d] = dist[c] + 1
                    q.append(d)
        for t in seats[s_idx + 1:]:
            dt = int(dist[int(t)])
            if dt < 0:
                return -1
            if dt > worst:
                worst = dt
    return worst
