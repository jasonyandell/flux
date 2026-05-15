"""Tiny arena tests for the v2 edge-flow heuristic."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import (
    ACTION_CLEAR_BASE,
    ACTION_NOOP,
    ACTION_SET_BASE,
    DEAD,
    K,
    MAX_STRENGTH,
    NEUTRAL,
    State,
    apply_actions,
    tick,
)
from flux_v2.edge_flow import (
    EdgeCategory,
    build_edge_flow_features,
    edge_flow_actions,
)


def _tiny_state(
    neighbors: list[list[int]],
    owner: list[int],
    strength: list[float],
    outflow: list[tuple[int, int]] | None = None,
    num_players: int = 2,
) -> State:
    n = len(owner)
    nb = np.full((n, K), -1, dtype=np.int32)
    for c, slots in enumerate(neighbors):
        for k, d in enumerate(slots[:K]):
            nb[c, k] = d
    flows = np.zeros((n, K), dtype=np.bool_)
    for c, k in outflow or []:
        flows[c, k] = True
    return State(
        N=n,
        pos=np.zeros((n, 2), dtype=np.float32),
        coord=np.zeros((n, 2), dtype=np.int32),
        neighbors=nb,
        owner=np.array(owner, dtype=np.int32),
        strength=np.array(strength, dtype=np.float32),
        outflow=flows,
        edge_pressure=np.zeros((n, K), dtype=np.float32),
        tick=0,
        num_players=num_players,
    )


def _set(slot: int) -> int:
    return ACTION_SET_BASE + slot


def _clear(slot: int) -> int:
    return ACTION_CLEAR_BASE + slot


def test_chain_sets_relay_toward_frontier():
    # 0 -> 1 -> neutral frontier. Cell 1 is already an active relay, so cell
    # 0 should connect to it instead of sitting as a max-strength back cell.
    s = _tiny_state(
        neighbors=[
            [1, -1, -1, -1, -1, -1],
            [2, -1, -1, 0, -1, -1],
            [-1, -1, -1, 1, -1, -1],
        ],
        owner=[0, 0, NEUTRAL],
        strength=[MAX_STRENGTH, MAX_STRENGTH, 10.0],
        outflow=[(1, 0)],
    )

    actions = edge_flow_actions(s, seat=0)

    assert actions[0] == _set(0)


def test_loop_holds_existing_local_relays():
    # A closed friendly loop has no frontier. Active slots point to friendly
    # relay cells, so the heuristic should preserve the staged pulse path.
    s = _tiny_state(
        neighbors=[
            [1, -1, -1, -1, -1, 2],
            [-1, 2, -1, 0, -1, -1],
            [0, -1, -1, -1, 1, -1],
        ],
        owner=[0, 0, 0],
        strength=[MAX_STRENGTH, MAX_STRENGTH, MAX_STRENGTH],
        outflow=[(0, 0), (1, 1), (2, 0)],
    )

    actions = edge_flow_actions(s, seat=0)

    assert actions.tolist() == [ACTION_NOOP, ACTION_NOOP, ACTION_NOOP]


def test_sink_trap_clears_friendly_max_dead_end():
    s = _tiny_state(
        neighbors=[
            [1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1],
        ],
        owner=[0, 0],
        strength=[MAX_STRENGTH, MAX_STRENGTH],
        outflow=[(0, 0)],
    )

    features = build_edge_flow_features(s, seat=0)
    actions = edge_flow_actions(s, seat=0)

    assert EdgeCategory(features.category[0, 0]) == EdgeCategory.MINE_TO_FRIENDLY_SINK
    assert actions[0] == _clear(0)


def test_weak_enemy_finish_beats_neutral_expansion_choice():
    s = _tiny_state(
        neighbors=[
            [1, 2, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1],
            [-1, -1, -1, -1, 0, -1],
        ],
        owner=[0, 1, NEUTRAL],
        strength=[MAX_STRENGTH, 8.0, 10.0],
    )

    actions = edge_flow_actions(s, seat=0)

    assert actions[0] == _set(0)


def test_neutral_race_prefers_productive_expansion_over_fill():
    s = _tiny_state(
        neighbors=[
            [1, 2, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1],
            [-1, -1, -1, -1, 0, -1],
        ],
        owner=[0, NEUTRAL, 0],
        strength=[MAX_STRENGTH, 10.0, 40.0],
    )

    actions = edge_flow_actions(s, seat=0)

    assert actions[0] == _set(0)


def test_two_fronts_open_independently():
    s = _tiny_state(
        neighbors=[
            [2, -1, -1, -1, -1, -1],
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        owner=[0, 0, NEUTRAL, 1],
        strength=[MAX_STRENGTH, MAX_STRENGTH, 10.0, 8.0],
    )

    actions = edge_flow_actions(s, seat=0)

    assert actions[0] == _set(0)
    assert actions[1] == _set(1)


def test_clears_blocked_dead_active_slot():
    s = _tiny_state(
        neighbors=[
            [1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1],
        ],
        owner=[0, DEAD],
        strength=[MAX_STRENGTH, 0.0],
        outflow=[(0, 0)],
    )

    actions = edge_flow_actions(s, seat=0)

    assert actions[0] == _clear(0)


Policy = Callable[[State, np.random.Generator], np.ndarray]


def _finish_arena() -> State:
    return _tiny_state(
        neighbors=[
            [2, -1, -1, -1, -1, -1],
            [-1, 2, -1, -1, -1, -1],
            [-1, -1, -1, 0, 1, -1],
        ],
        owner=[0, 0, 1],
        strength=[MAX_STRENGTH, MAX_STRENGTH, 0.4],
    )


def _run_policy(state: State, policy: Policy, seed: int, steps: int = 6) -> State:
    rng = np.random.default_rng(seed)
    s = state
    for _ in range(steps):
        s = apply_actions(s, policy(s, rng))
        s = tick(s)
    return s


def _owned_score(state: State, seat: int = 0) -> int:
    return int((state.owner == seat).sum())


def _edge_policy(state: State, rng: np.random.Generator) -> np.ndarray:
    return edge_flow_actions(state, seat=0, rng=rng)


def _inert_policy(state: State, rng: np.random.Generator) -> np.ndarray:
    return np.full(state.N, ACTION_NOOP, dtype=np.int32)


def _random_policy(state: State, rng: np.random.Generator) -> np.ndarray:
    actions = np.full(state.N, ACTION_NOOP, dtype=np.int32)
    owned = np.where(state.owner == 0)[0]
    if owned.size:
        actions[owned] = rng.integers(0, ACTION_NOOP + 1, size=owned.size)
    return actions


def test_edge_flow_beats_inert_and_random_on_finish_arena():
    start = _finish_arena()

    edge_score = _owned_score(_run_policy(start, _edge_policy, seed=0))
    inert_score = _owned_score(_run_policy(start, _inert_policy, seed=0))
    random_scores = [
        _owned_score(_run_policy(start, _random_policy, seed=i))
        for i in range(32)
    ]

    assert edge_score > inert_score
    assert edge_score > float(np.mean(random_scores))
