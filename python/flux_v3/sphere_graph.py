"""Geodesic-sphere graph for v2.

Subdivided icosahedron — vertices are cells, edges of the triangulation are
adjacencies. Most cells have 6 neighbors; the 12 original icosahedron
vertices have 5 (pentagonal), with their 6th neighbor slot left at -1.

Slot ordering on the sphere can't satisfy `OPPOSITE_SLOT[k] = (k+3) % 6`
globally — adjacent tangent frames don't agree well enough to make the
+3 invariant hold across every edge. So the builder picks an arbitrary
slot order per cell (angular sort with a globally-consistent reference
just for visual nicety) and ships a per-cell `back_slot[c, k]` lookup
table to the reducer via the State field. Both step.py and the lightning
solver consume this — see `back_slot_or_default()`.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .state import (
    DEAD,
    K,
    NEUTRAL,
    State,
    compute_back_slot,
)


# ---------------------------------------------------------------------------
# Subdivided icosahedron
# ---------------------------------------------------------------------------

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _icosahedron() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """12 vertices on the unit sphere, 20 triangular faces."""
    verts = np.array(
        [
            [-1,  PHI, 0], [ 1,  PHI, 0], [-1, -PHI, 0], [ 1, -PHI, 0],
            [0, -1,  PHI], [0,  1,  PHI], [0, -1, -PHI], [0,  1, -PHI],
            [ PHI, 0, -1], [ PHI, 0,  1], [-PHI, 0, -1], [-PHI, 0,  1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return verts, faces


def build_icosphere(subdiv: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Subdivide an icosahedron `subdiv` times. Each face becomes 4^subdiv
    triangles; vertex count is 10·4^subdiv + 2.

    Returns (vertices (V,3) float64, edges set of (i, j) with i<j).
    """
    verts, faces = _icosahedron()
    verts_list = [tuple(v) for v in verts.tolist()]
    vert_index: dict[tuple[float, float, float], int] = {v: i for i, v in enumerate(verts_list)}

    def midpoint(i: int, j: int) -> int:
        a = np.asarray(verts_list[i])
        b = np.asarray(verts_list[j])
        m = (a + b) / 2.0
        m = m / np.linalg.norm(m)
        key = tuple(m.tolist())
        if key in vert_index:
            return vert_index[key]
        idx = len(verts_list)
        verts_list.append(key)
        vert_index[key] = idx
        return idx

    for _ in range(subdiv):
        new_faces: list[tuple[int, int, int]] = []
        for (a, b, c) in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append((a, ab, ca))
            new_faces.append((b, bc, ab))
            new_faces.append((c, ca, bc))
            new_faces.append((ab, bc, ca))
        faces = new_faces

    # Edge set, undirected.
    edges: set[tuple[int, int]] = set()
    for (a, b, c) in faces:
        for (i, j) in ((a, b), (b, c), (c, a)):
            edges.add((min(i, j), max(i, j)))

    return np.asarray(verts_list, dtype=np.float64), edges


# ---------------------------------------------------------------------------
# Slot assignment
# ---------------------------------------------------------------------------

# Global reference for tangent-frame construction. Most cells use Y-up;
# cells whose normal is too close to Y-up fall back to Z-up so the projection
# never degenerates.
_GLOBAL_UP = np.array([0.0, 1.0, 0.0])
_GLOBAL_FALLBACK = np.array([0.0, 0.0, 1.0])
_POLE_TOL = 0.95   # |n · up| above this → use fallback


def _tangent_frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (e_x, e_y) tangent basis at a unit-normal n. Globally consistent
    so that adjacent cells get nearly-parallel frames — required for
    OPPOSITE_SLOT to hold."""
    up = _GLOBAL_UP if abs(np.dot(n, _GLOBAL_UP)) < _POLE_TOL else _GLOBAL_FALLBACK
    e_y = up - np.dot(up, n) * n
    e_y = e_y / np.linalg.norm(e_y)
    e_x = np.cross(n, e_y)
    return e_x, e_y


def _assign_slots(
    verts: np.ndarray, neighbors_unsorted: list[list[int]],
) -> np.ndarray:
    """Pack each cell's neighbors into the K slots in tangent-angle order.

    With the per-cell `back_slot` lookup table the reducer no longer needs
    OPPOSITE_SLOT-style globally-consistent slot ordering, so this just sorts
    each cell's neighbors angularly (purely cosmetic — it gives slot 0 a
    consistent "global up" bias, which makes flow arrows render with a
    sensible orientation) and packs them densely from slot 0. Pentagonal
    cells leave slot 5 at -1.
    """
    V = verts.shape[0]
    nb = np.full((V, K), -1, dtype=np.int32)
    for c in range(V):
        n = verts[c]
        e_x, e_y = _tangent_frame(n)
        scored: list[tuple[float, int]] = []
        for d in neighbors_unsorted[c]:
            t = verts[d] - n
            t = t - np.dot(t, n) * n
            angle = math.atan2(np.dot(t, e_y), np.dot(t, e_x))
            scored.append((angle, d))
        scored.sort()
        for slot, (_, d) in enumerate(scored):
            if slot >= K:
                raise RuntimeError(f"cell {c} has degree > {K}")
            nb[c, slot] = d
    return nb


# ---------------------------------------------------------------------------
# Board builder
# ---------------------------------------------------------------------------

def make_sphere_board(
    subdiv: int,
    num_players: int,
    seat_cells: Optional[np.ndarray] = None,
    dead_cells: Optional[np.ndarray] = None,
    seat_strength: float = 30.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[State, np.ndarray]:
    """Build a sphere-topology v2 board.

    Returns (state, pos3d) where pos3d (V, 3) float32 is the 3D vertex
    positions on the unit sphere. The State's `pos` field carries a 2D
    sphere-azimuth/colatitude pair for compatibility with code that assumes
    a flat layout, but real geometry lives in `pos3d` and is shipped to the
    viewer via replay metadata.
    """
    verts, edges = build_icosphere(subdiv)
    V = verts.shape[0]

    neighbors_unsorted: list[list[int]] = [[] for _ in range(V)]
    for (i, j) in edges:
        neighbors_unsorted[i].append(j)
        neighbors_unsorted[j].append(i)

    neighbors = _assign_slots(verts, neighbors_unsorted)
    back_slot = compute_back_slot(neighbors)

    pos3d = verts.astype(np.float32)

    # 2D pos: azimuth/colatitude — solvers don't use it for sphere boards but
    # the State dataclass wants something here.
    pos = np.zeros((V, 2), dtype=np.float32)
    for i in range(V):
        x, y, z = verts[i]
        pos[i, 0] = math.atan2(y, x)
        pos[i, 1] = math.acos(max(-1.0, min(1.0, z)))
    coord = np.zeros((V, 2), dtype=np.int32)

    owner = np.full(V, NEUTRAL, dtype=np.int32)
    strength = np.full(V, 10.0, dtype=np.float32)

    # Default seat layout: spread `num_players` cells maximally apart on the
    # sphere using a simple greedy farthest-point on the live-cell set.
    if seat_cells is None:
        if rng is None:
            rng = np.random.default_rng(0)
        live_pool = np.arange(V) if dead_cells is None or len(dead_cells) == 0 else (
            np.setdiff1d(np.arange(V), dead_cells, assume_unique=False)
        )
        # Start from a random live cell; greedy farthest-point in great-circle
        # distance keeps seats spread out without solving an n-body problem.
        first = int(rng.choice(live_pool))
        chosen = [first]
        for _ in range(num_players - 1):
            best, best_d = -1, -1.0
            for cand in live_pool:
                if cand in chosen:
                    continue
                # Min angular distance from cand to any chosen seat.
                p = pos3d[cand]
                d_min = min(
                    math.acos(max(-1.0, min(1.0, float(np.dot(p, pos3d[s])))))
                    for s in chosen
                )
                if d_min > best_d:
                    best_d = d_min
                    best = int(cand)
            chosen.append(best)
        seat_cells = np.asarray(chosen, dtype=np.int32)

    if dead_cells is not None and len(dead_cells) > 0:
        owner[dead_cells] = DEAD
        strength[dead_cells] = 0.0

    for p, cell in enumerate(seat_cells[:num_players]):
        c = int(cell)
        if owner[c] == DEAD:
            raise ValueError(f"seat {p} placed on a dead cell ({c})")
        owner[c] = p
        strength[c] = seat_strength

    outflow = np.zeros((V, K), dtype=np.bool_)
    edge_pressure = np.zeros((V, K), dtype=np.float32)

    state = State(
        N=V,
        pos=pos,
        coord=coord,
        neighbors=neighbors,
        owner=owner,
        strength=strength,
        outflow=outflow,
        edge_pressure=edge_pressure,
        tick=0,
        num_players=num_players,
        back_slot=back_slot,
    )
    return state, pos3d
