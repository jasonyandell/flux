"""Regen-flow step semantics.

Outflow has no health cost; instead a cell with outflows forfeits its own
regen, redirected as flux split evenly across the outflows. Damage is
symmetric (no attack bonus). Capture flips ownership and resets strength
to CAPTURE_STRENGTH (=1.0); any leftover damage past 0 is wasted.

This is a parallel implementation to step.py — neither path imports from
the other. The browser-side mirror lives at src/game/step_regen.ts.
"""
from __future__ import annotations
from typing import List, Optional

from .state import (
    Action,
    Edge,
    Flow,
    GameNode,
    GameState,
    MAX_STRENGTH,
)

REGEN_BASE_PER_SEC: float = 0.5
REGEN_SLOPE: float = 2.0
CAPTURE_STRENGTH: float = 1.0


def regen_rate(strength: float) -> float:
    return REGEN_BASE_PER_SEC * (1.0 + REGEN_SLOPE * (strength - 1.0))


_adjacency_cache: dict[int, set[int]] = {}


def _edge_key(a: int, b: int) -> int:
    return a * 10_000_000 + b if a < b else b * 10_000_000 + a


def _adjacency_set(edges: tuple[Edge, ...]) -> set[int]:
    k = id(edges)
    s = _adjacency_cache.get(k)
    if s is not None:
        return s
    s = set()
    for e in edges:
        s.add(_edge_key(e.a, e.b))
    _adjacency_cache[k] = s
    return s


def apply_action(state: GameState, action: Action) -> GameState:
    if action.kind != "toggleFlow":
        return state
    if action.src < 0 or action.src >= len(state.nodes):
        return state
    src = state.nodes[action.src]
    if src.owner != action.player:
        return state
    if _edge_key(action.src, action.dst) not in _adjacency_set(state.edges):
        return state
    on_edge = -1
    for i, f in enumerate(state.flows):
        if (f.src == action.src and f.dst == action.dst) or (
            f.src == action.dst and f.dst == action.src
        ):
            on_edge = i
            break
    if on_edge < 0:
        return GameState(
            nodes=state.nodes,
            edges=state.edges,
            flows=state.flows + (Flow(src=action.src, dst=action.dst, player=action.player),),
            tick=state.tick,
            num_players=state.num_players,
        )
    existing = state.flows[on_edge]
    exact = (
        existing.src == action.src
        and existing.dst == action.dst
        and existing.player == action.player
    )
    flows = tuple(f for j, f in enumerate(state.flows) if j != on_edge)
    if exact:
        return GameState(
            nodes=state.nodes,
            edges=state.edges,
            flows=flows,
            tick=state.tick,
            num_players=state.num_players,
        )
    return GameState(
        nodes=state.nodes,
        edges=state.edges,
        flows=flows + (Flow(src=action.src, dst=action.dst, player=action.player),),
        tick=state.tick,
        num_players=state.num_players,
    )


def step_regen(state: GameState, dt: float) -> GameState:
    N = state.num_players
    n_nodes = len(state.nodes)
    forces: List[List[float]] = [[0.0] * N for _ in range(n_nodes)]

    # Count friendly outflows per cell.
    out_count: List[int] = [0] * n_nodes
    for flow in state.flows:
        src = state.nodes[flow.src]
        if src.owner != flow.player:
            continue
        out_count[flow.src] += 1

    # Idle cells regen normally; sending cells forfeit their regen (it's
    # redirected to outflows below).
    for node in state.nodes:
        if node.owner is None:
            continue
        if out_count[node.id] == 0:
            forces[node.id][node.owner] += regen_rate(node.strength) * dt

    # Outflow contributions: each delivers regen(src) / K. Friendly destinations
    # gain it; enemy destinations lose it (symmetric damage).
    alive_flows: List[Flow] = []
    for flow in state.flows:
        src = state.nodes[flow.src]
        if src.owner != flow.player:
            continue
        alive_flows.append(flow)
        flux = (regen_rate(src.strength) * dt) / out_count[flow.src]
        dst = state.nodes[flow.dst]
        if dst.owner == flow.player:
            forces[flow.dst][flow.player] += flux         # support
        else:
            forces[flow.dst][flow.player] -= flux         # attack (symmetric)

    # Resolve. force[p] is signed: positive = owner-side support, negative =
    # attack by player p (magnitude -force[p]). Sum across all p to get the
    # net strength delta; capture flips on net < 0.
    new_nodes: List[GameNode] = []
    for node in state.nodes:
        force = forces[node.id]
        owner: Optional[int] = node.owner
        strength = node.strength

        for p in range(N):
            strength += force[p]

        if strength < 0:
            best_p: Optional[int] = None
            best_damage = 0.0
            for p in range(N):
                if p != owner and -force[p] > best_damage:
                    best_damage = -force[p]
                    best_p = p
            if best_p is not None:
                owner = best_p
                strength = CAPTURE_STRENGTH
            else:
                strength = 0.0

        if strength > MAX_STRENGTH:
            strength = MAX_STRENGTH
        if strength < 0:
            strength = 0.0

        new_nodes.append(GameNode(id=node.id, pos=node.pos, owner=owner, strength=strength))

    return GameState(
        nodes=tuple(new_nodes),
        edges=state.edges,
        flows=tuple(alive_flows),
        tick=state.tick + 1,
        num_players=state.num_players,
    )
