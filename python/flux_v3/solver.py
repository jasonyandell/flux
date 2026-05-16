"""Old-school algorithmic solver for flux v2.

A hand-written greedy local heuristic — no neural net. For each owned cell
the solver computes an "ideal" outflow set from local neighbor classification
plus a BFS distance-to-frontier map, then emits one action that nudges the
cell's current outflow toward that set.

Per-cell ideal outflow set for seat `s` at cell `c`:

  attack/expand   slot points at enemy or neutral (not dead, not off-grid)
  relay           slot points at friendly `d` with dist_to_frontier[d] < dist[c]
  excluded        everything else (interior friendlies with no closer front,
                  off-grid, dead cells, dead-end MAX sinks)

A "dead-end MAX sink" is a friendly neighbor at MAX strength with zero
active outflows — same definition the reward's dest-terminated waste uses.

Action priority per cell (one action per AI tick):
  1. SET a missing attack/expand slot
  2. SET a missing relay slot
  3. CLEAR a stale slot (currently on but not in ideal set)
  4. NOOP

Board precondition (see `wiki/decisions/v2-board-connectivity.md`): every
non-DEAD cell must be reachable from every other through non-DEAD neighbors,
and every seat must be reachable from every other seat. The
distance-to-frontier BFS only traverses owned cells, but the algorithm
implicitly assumes the live subgraph is one connected component — an
isolated live pocket sits at the `BIG` sentinel distance forever and only
emits attack actions on its own boundary, never relays. The solver runner
enforces this via `seats_mutually_reachable` + max-pair-distance retry, or
the `carve` mode that revives dead cells along bridges.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .state import (
    ACTION_CLEAR_BASE,
    ACTION_NOOP,
    ACTION_SET_BASE,
    DEAD,
    K,
    MAX_STRENGTH,
    NEUTRAL,
    State,
)


def _frontier_distance(
    state: State, seat: int,
) -> np.ndarray:
    """BFS over seat-owned cells from the set of cells adjacent to any
    non-friendly (enemy/neutral/dead-free) tile. Returns (N,) int32 distances;
    cells not owned by `seat` or unreachable get a large sentinel.

    "Non-friendly" includes neutral and enemy. Dead cells are walls — they do
    not act as sources of distance-0 and they do not get traversed.
    """
    N = state.N
    owner = state.owner
    nb = state.neighbors

    BIG = np.int32(10_000)
    dist = np.full(N, BIG, dtype=np.int32)

    is_mine = (owner == seat)
    if not is_mine.any():
        return dist

    # Seed: any owned cell with at least one non-friendly *valid* neighbor.
    queue: deque[int] = deque()
    for c in np.where(is_mine)[0]:
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            # Non-friendly + traversable target: enemy seats and NEUTRAL.
            # DEAD cells block flow, so they don't count as frontier targets.
            if od != seat and od != DEAD:
                dist[c] = 0
                queue.append(int(c))
                break

    # Standard BFS through owned cells only.
    while queue:
        c = queue.popleft()
        dc = dist[c]
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            if owner[d] != seat:
                continue
            if dist[d] <= dc + 1:
                continue
            dist[d] = dc + 1
            queue.append(d)

    return dist


def _ideal_outflow_for_cell(
    c: int,
    seat: int,
    state: State,
    dist: np.ndarray,
    num_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (attack_mask, relay_mask) — bool arrays of length K.

    `attack_mask[k]` is True iff slot k points at an enemy or neutral cell
    (good target to set).

    `relay_mask[k]` is True iff slot k points at a friendly cell that's
    strictly closer to the frontier than `c`.

    Their union is the cell's ideal-on set. Slots not in either should be
    cleared if currently on.
    """
    nb = state.neighbors
    owner = state.owner
    strength = state.strength

    attack = np.zeros(K, dtype=np.bool_)
    relay = np.zeros(K, dtype=np.bool_)

    dc = dist[c]
    for k in range(K):
        d = int(nb[c, k])
        if d < 0:
            continue
        od = int(owner[d])
        if od == DEAD:
            continue
        if od != seat:
            # Enemy or neutral — always a good attack/expansion target.
            attack[k] = True
            continue
        # Friendly neighbor.
        # Dead-end MAX sink — don't pump pressure into it.
        if strength[d] >= MAX_STRENGTH and num_active[d] == 0:
            continue
        # Relay only if the neighbor is strictly closer to the frontier.
        if dist[d] < dc:
            relay[k] = True
    return attack, relay


def solver_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute (N,) int32 actions for seat `seat`.

    Cells not owned by `seat` get NOOP. Owned cells get the one action that
    most improves their outflow toward the ideal set (priority listed in the
    module docstring).

    `rng` is used only to break ties between equally-attractive slots when
    multiple are available; if None, the lowest-index slot wins.
    """
    N = state.N
    owner = state.owner
    outflow = state.outflow

    actions = np.full(N, ACTION_NOOP, dtype=np.int32)

    is_mine = (owner == seat)
    if not is_mine.any():
        return actions

    dist = _frontier_distance(state, seat)
    num_active = outflow.sum(axis=1).astype(np.int32)

    owned_cells = np.where(is_mine)[0]
    for c in owned_cells:
        c = int(c)
        attack, relay = _ideal_outflow_for_cell(c, seat, state, dist, num_active)
        cur = outflow[c]

        # 1. Missing attack slot (set).
        missing_attack = np.where(attack & ~cur)[0]
        if missing_attack.size:
            k = _pick(missing_attack, rng)
            actions[c] = ACTION_SET_BASE + k
            continue

        # 2. Missing relay slot (set).
        missing_relay = np.where(relay & ~cur)[0]
        if missing_relay.size:
            k = _pick(missing_relay, rng)
            actions[c] = ACTION_SET_BASE + k
            continue

        # 3. Stale slot to clear: currently on but not in attack ∪ relay.
        ideal = attack | relay
        stale = np.where(cur & ~ideal)[0]
        if stale.size:
            k = _pick(stale, rng)
            actions[c] = ACTION_CLEAR_BASE + k
            continue

        # 4. Otherwise NOOP — outflow already matches the ideal set.

    return actions


def _pick(candidates: np.ndarray, rng: Optional[np.random.Generator]) -> int:
    if rng is None:
        return int(candidates[0])
    return int(candidates[rng.integers(candidates.size)])
