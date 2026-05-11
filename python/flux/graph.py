from __future__ import annotations
import math
from typing import List, Tuple
from .state import GameNode, Edge, GameState, Vec2

SQRT3 = math.sqrt(3.0)


def axial_to_pixel(q: int, r: int) -> Vec2:
    return Vec2(SQRT3 * q + (SQRT3 / 2.0) * r, (3.0 / 2.0) * r)


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    dq, dr = q1 - q2, r1 - r2
    return max(abs(dq), abs(dr), abs(dq + dr))


def make_initial_state(radius: int = 18, distance: int = 2, num_players: int = 6) -> GameState:
    cells: List[Tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            cells.append((q, r))

    id_by_key: dict[Tuple[int, int], int] = {}
    for i, c in enumerate(cells):
        id_by_key[c] = i

    nodes: List[GameNode] = []
    for i, (q, r) in enumerate(cells):
        p = axial_to_pixel(q, r)
        nodes.append(GameNode(id=i, pos=p, owner=None, strength=10.0))

    perimeter_ids: List[int] = []
    for i, (q, r) in enumerate(cells):
        if max(abs(q), abs(r), abs(q + r)) == radius:
            perimeter_ids.append(i)

    # JS Array.prototype.sort is not stable across all engines historically but
    # since V8 7.0 (Chrome 70) it's stable. The compare returns a float diff;
    # Python's sort is stable and we'll match by using the same float key.
    perimeter_ids.sort(key=lambda i: math.atan2(nodes[i].pos.y, nodes[i].pos.x))

    # Reassign perimeter owners. Build a mutable copy of nodes.
    mutable_nodes = list(nodes)
    for p in range(num_players):
        idx = perimeter_ids[(p * len(perimeter_ids)) // num_players]
        n = mutable_nodes[idx]
        mutable_nodes[idx] = GameNode(id=n.id, pos=n.pos, owner=p, strength=30.0)

    edges: List[Edge] = []
    for i, (q, r) in enumerate(cells):
        for dq in range(-distance, distance + 1):
            for dr in range(-distance, distance + 1):
                if dq == 0 and dr == 0:
                    continue
                if max(abs(dq), abs(dr), abs(dq + dr)) > distance:
                    continue
                j = id_by_key.get((q + dq, r + dr))
                if j is None or j <= i:
                    continue
                edges.append(Edge(a=i, b=j))

    return GameState(
        nodes=tuple(mutable_nodes),
        edges=tuple(edges),
        flows=tuple(),
        tick=0,
        num_players=num_players,
    )
