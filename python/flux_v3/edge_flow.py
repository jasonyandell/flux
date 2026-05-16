"""Non-learning edge-flow heuristic for flux v2.

This module is intentionally separate from the older solver experiments. It
derives edge categories from the current state, scores local route usefulness,
then emits exactly one v2 action per owned cell: Set one slot, Clear one slot,
or No-op.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from .state import (
    ACTION_CLEAR_BASE,
    ACTION_NOOP,
    ACTION_SET_BASE,
    CAPTURE_STRENGTH,
    DEAD,
    K,
    MAX_STRENGTH,
    NEUTRAL,
    State,
)


class EdgeCategory(IntEnum):
    """Directed edge category from one acting seat's point of view."""

    BLOCKED = 0
    MINE_TO_ENEMY = 1
    MINE_TO_NEUTRAL = 2
    MINE_TO_FRIENDLY_RELAY = 3
    MINE_TO_FRIENDLY_SINK = 4
    MINE_TO_FRIENDLY_FILL = 5
    ENEMY_TO_MINE = 6
    ENEMY_TO_ENEMY = 7
    OTHER = 8


@dataclass(frozen=True)
class EdgeFlowFeatures:
    """Derived local-flow facts used by the heuristic and tests."""

    category: np.ndarray
    score: np.ndarray
    frontier_distance: np.ndarray
    active_outflows: np.ndarray


BIG_DISTANCE = np.int32(10_000)


def build_edge_flow_features(
    state: State,
    seat: int,
    *,
    open_strength: float = MAX_STRENGTH - 1e-4,
) -> EdgeFlowFeatures:
    """Build `(N, K)` edge categories and intent scores for `seat`.

    Positive scores mean the edge is useful to open or keep open. Negative
    scores mean a currently-active slot is stale and should be cleared. Near
    zero leaves the current gate alone, preserving local hold/release timing
    instead of imposing a global pulse.
    """
    N = state.N
    owner = state.owner
    strength = state.strength
    nb = state.neighbors
    outflow = state.outflow

    active = outflow.sum(axis=1).astype(np.int32)
    dist = _frontier_distance(state, seat)

    category = np.full((N, K), EdgeCategory.BLOCKED, dtype=np.int8)
    score = np.zeros((N, K), dtype=np.float32)

    for c in range(N):
        src_owner = int(owner[c])
        for k in range(K):
            d = int(nb[c, k])
            if d < 0 or src_owner == DEAD:
                category[c, k] = EdgeCategory.BLOCKED
                score[c, k] = -100.0
                continue

            dst_owner = int(owner[d])
            if dst_owner == DEAD:
                category[c, k] = EdgeCategory.BLOCKED
                score[c, k] = -100.0
                continue

            if src_owner == seat:
                cat = _mine_edge_category(state, seat, d, active)
                category[c, k] = cat
                score[c, k] = _mine_edge_score(
                    state=state,
                    src=c,
                    dst=d,
                    category=cat,
                    dist=dist,
                    active=active,
                    open_strength=open_strength,
                )
            elif src_owner >= 0:
                if dst_owner == seat:
                    category[c, k] = EdgeCategory.ENEMY_TO_MINE
                elif dst_owner >= 0:
                    category[c, k] = EdgeCategory.ENEMY_TO_ENEMY
                else:
                    category[c, k] = EdgeCategory.OTHER
            else:
                category[c, k] = EdgeCategory.OTHER

    return EdgeFlowFeatures(
        category=category,
        score=score,
        frontier_distance=dist,
        active_outflows=active,
    )


def edge_flow_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
    *,
    open_threshold: float = 30.0,
    clear_threshold: float = 0.0,
    open_strength: float = MAX_STRENGTH - 1e-4,
) -> np.ndarray:
    """Return one v2 action per cell for `seat`.

    The action selector is deliberately local: open the best missing productive
    edge, otherwise clear the worst stale active edge, otherwise hold.
    """
    actions = np.full(state.N, ACTION_NOOP, dtype=np.int32)
    mine = state.owner == seat
    if not mine.any():
        return actions

    features = build_edge_flow_features(
        state,
        seat,
        open_strength=open_strength,
    )

    for c in np.where(mine)[0]:
        c = int(c)
        cur = state.outflow[c]
        scores = features.score[c]

        missing = np.where((~cur) & (scores >= open_threshold))[0]
        if missing.size:
            k = _pick_by_score(missing, scores, rng, highest=True)
            actions[c] = ACTION_SET_BASE + k
            continue

        stale = np.where(cur & (scores <= clear_threshold))[0]
        if stale.size:
            k = _pick_by_score(stale, scores, rng, highest=False)
            actions[c] = ACTION_CLEAR_BASE + k

    return actions


def _mine_edge_category(
    state: State,
    seat: int,
    dst: int,
    active: np.ndarray,
) -> EdgeCategory:
    dst_owner = int(state.owner[dst])
    if dst_owner >= 0 and dst_owner != seat:
        return EdgeCategory.MINE_TO_ENEMY
    if dst_owner == NEUTRAL:
        return EdgeCategory.MINE_TO_NEUTRAL
    if dst_owner == seat:
        if active[dst] > 0:
            return EdgeCategory.MINE_TO_FRIENDLY_RELAY
        if state.strength[dst] >= MAX_STRENGTH:
            return EdgeCategory.MINE_TO_FRIENDLY_SINK
        return EdgeCategory.MINE_TO_FRIENDLY_FILL
    return EdgeCategory.OTHER


def _mine_edge_score(
    *,
    state: State,
    src: int,
    dst: int,
    category: EdgeCategory,
    dist: np.ndarray,
    active: np.ndarray,
    open_strength: float,
) -> float:
    ready = float(state.strength[src]) >= open_strength

    if category == EdgeCategory.MINE_TO_ENEMY:
        if not ready:
            return 10.0
        dst_strength = float(state.strength[dst])
        weak_bonus = max(0.0, MAX_STRENGTH - dst_strength) / MAX_STRENGTH
        finish_bonus = 25.0 if dst_strength <= CAPTURE_STRENGTH else 0.0
        return 90.0 + 30.0 * weak_bonus + finish_bonus

    if category == EdgeCategory.MINE_TO_NEUTRAL:
        if not ready:
            return 8.0
        headroom = max(0.0, MAX_STRENGTH - float(state.strength[dst]))
        return 70.0 + 10.0 * (headroom / MAX_STRENGTH)

    if category == EdgeCategory.MINE_TO_FRIENDLY_RELAY:
        # Keep existing local loops alive even when no frontier exists. Missing
        # relay gates only open when they route strictly closer to a frontier.
        if dist[dst] < dist[src]:
            return 45.0 + min(15.0, float(dist[src] - dist[dst]) * 5.0)
        return 8.0 if active[dst] > 0 else 0.0

    if category == EdgeCategory.MINE_TO_FRIENDLY_FILL:
        if dist[dst] < dist[src] and ready:
            headroom = max(0.0, MAX_STRENGTH - float(state.strength[dst]))
            return 30.0 + 10.0 * (headroom / MAX_STRENGTH)
        return 6.0 if float(state.strength[dst]) < MAX_STRENGTH else 0.0

    if category == EdgeCategory.MINE_TO_FRIENDLY_SINK:
        return -80.0

    if category == EdgeCategory.BLOCKED:
        return -100.0

    return 0.0


def _frontier_distance(state: State, seat: int) -> np.ndarray:
    """Distance through owned cells to the nearest enemy/neutral frontier."""
    owner = state.owner
    nb = state.neighbors
    dist = np.full(state.N, BIG_DISTANCE, dtype=np.int32)
    queue: deque[int] = deque()

    for c in np.where(owner == seat)[0]:
        c = int(c)
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            dst_owner = int(owner[d])
            if dst_owner != seat and dst_owner != DEAD:
                dist[c] = 0
                queue.append(c)
                break

    while queue:
        c = queue.popleft()
        next_dist = np.int32(dist[c] + 1)
        for k in range(K):
            d = int(nb[c, k])
            if d < 0 or owner[d] != seat:
                continue
            if dist[d] <= next_dist:
                continue
            dist[d] = next_dist
            queue.append(d)

    return dist


def _pick_by_score(
    candidates: np.ndarray,
    scores: np.ndarray,
    rng: Optional[np.random.Generator],
    *,
    highest: bool,
) -> int:
    values = scores[candidates]
    best = values.max() if highest else values.min()
    tied = candidates[np.isclose(values, best)]
    if rng is None:
        return int(tied[0])
    return int(tied[rng.integers(tied.size)])
