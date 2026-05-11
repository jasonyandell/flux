from __future__ import annotations
from typing import List, Optional
from .state import (
    Action,
    Edge,
    Flow,
    GameNode,
    GameState,
    ATTACK_BONUS,
    MAX_STRENGTH,
    MIN_STRENGTH_TO_SEND,
    REGEN_PER_SEC,
    TRANSFER_PER_SEC,
)

# Per-edges-tuple adjacency cache, keyed by id() (object identity), like the
# JS WeakMap keyed on the edges array.
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


def step(state: GameState, dt: float) -> GameState:
    N = state.num_players
    n_nodes = len(state.nodes)
    forces: List[List[float]] = [[0.0] * N for _ in range(n_nodes)]

    for node in state.nodes:
        if node.owner is not None:
            forces[node.id][node.owner] += REGEN_PER_SEC * dt

    alive_flows: List[Flow] = []
    for flow in state.flows:
        src = state.nodes[flow.src]
        if src.owner != flow.player:
            continue
        alive_flows.append(flow)
        if src.strength < MIN_STRENGTH_TO_SEND:
            continue
        k = TRANSFER_PER_SEC * dt
        forces[flow.src][flow.player] -= k
        enemy = state.nodes[flow.dst].owner != flow.player
        forces[flow.dst][flow.player] += k * (1.0 + ATTACK_BONUS) if enemy else k

    new_nodes: List[GameNode] = []
    for node in state.nodes:
        force = forces[node.id]
        owner: Optional[int] = node.owner
        strength = node.strength

        if owner is not None:
            strength += force[owner]
            for p in range(N):
                if p != owner:
                    strength -= force[p]
        else:
            for p in range(N):
                if force[p] > 0:
                    strength -= force[p]

        if strength < 0:
            best_p: Optional[int] = None
            best = 0.0
            for p in range(N):
                if p != owner and force[p] > best:
                    best = force[p]
                    best_p = p
            if best_p is not None:
                owner = best_p
                strength = -strength
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
