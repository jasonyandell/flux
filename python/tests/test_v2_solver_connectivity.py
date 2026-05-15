"""Board-connectivity invariant for the v2 BFS and lightning solvers.

Both solvers assume the non-DEAD subgraph of the board is one connected
component AND every seat is reachable from every other seat through non-DEAD
neighbors. Isolated live pockets break the BFS distance-to-frontier in
`solver.py` and starve the potential field in `solver_lightning.py`. The
solver runner enforces this with a `retry` mode (regenerate boards until
`seats_mutually_reachable` + `max_seat_pair_distance ≤ 4·R`) or a `carve`
mode that revives dead cells along the shortest bridge. These tests pin
the invariant down across many seeds and both modes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from flux_v2 import DEAD, NEUTRAL, make_board, random_seat_and_dead
from flux_v2.graph import (
    _live_subgraph_connected,
    carve_seat_connectors,
    max_seat_pair_distance,
    seats_mutually_reachable,
)
from scripts.run_v2_solver import _build_initial_state


# --------------------- runner produces connected boards ---------------------


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize(
    "radius,num_players,num_dead",
    [
        (6, 6, 10),
        (6, 6, 40),
        (9, 12, 40),
    ],
)
def test_retry_mode_boards_are_connected(seed, radius, num_players, num_dead):
    """Every board from `_build_initial_state` (retry mode, the default) has a
    fully connected non-DEAD subgraph and every seat reachable from every
    other through non-DEAD neighbors."""
    rng = np.random.default_rng(seed)
    state, dead = _build_initial_state(
        radius, num_players, num_dead, rng, connect_mode="retry",
    )
    is_dead = state.owner == DEAD
    assert _live_subgraph_connected(is_dead, state.neighbors)
    seats = np.where((state.owner >= 0) & (state.owner != NEUTRAL))[0]
    assert seats_mutually_reachable(seats, dead, state.neighbors)
    worst = max_seat_pair_distance(seats, dead, state.neighbors)
    assert 0 <= worst <= max(4 * radius, 6)


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize(
    "radius,num_players,num_dead",
    [
        (6, 6, 30),
        (9, 12, 80),
    ],
)
def test_carve_mode_boards_are_connected(seed, radius, num_players, num_dead):
    """Carve mode starts from a uniform-random dead-cell sample (which may
    have isolated live pockets) and revives just enough dead cells to bridge
    them. After carving the live subgraph must be one connected component."""
    rng = np.random.default_rng(seed)
    state, dead = _build_initial_state(
        radius, num_players, num_dead, rng, connect_mode="carve",
    )
    is_dead = state.owner == DEAD
    assert _live_subgraph_connected(is_dead, state.neighbors)
    seats = np.where((state.owner >= 0) & (state.owner != NEUTRAL))[0]
    assert seats_mutually_reachable(seats, dead, state.neighbors)


# --------------------- helper-function unit tests ---------------------


def test_seats_mutually_reachable_rejects_islanded_seat():
    """A hand-built board with one seat walled off by DEAD cells fails the
    mutual-reachability check."""
    base = make_board(radius=3, num_players=2)
    is_dead = np.zeros(base.N, dtype=np.bool_)
    # Surround cell 0 with DEAD on every valid neighbor slot, isolating it.
    for k in range(base.neighbors.shape[1]):
        d = int(base.neighbors[0, k])
        if d >= 0:
            is_dead[d] = True
    dead = np.where(is_dead)[0].astype(np.int32)
    # Place a seat on cell 0 (the islanded cell) and another seat far away.
    far_seat = int(np.where(~is_dead)[0][-1])
    seats = np.array([0, far_seat], dtype=np.int32)
    assert not seats_mutually_reachable(seats, dead, base.neighbors)


def test_seats_mutually_reachable_accepts_no_dead_board():
    """A board with no dead cells is trivially mutually reachable."""
    base = make_board(radius=3, num_players=4)
    seats = np.array([0, 5, 10, base.N - 1], dtype=np.int32)
    dead = np.zeros(0, dtype=np.int32)
    assert seats_mutually_reachable(seats, dead, base.neighbors)


def test_carve_seat_connectors_bridges_islands():
    """A hand-constructed board with two seat-bearing components gets
    carved into one connected component, with carved cells lying on the
    revived bridge."""
    base = make_board(radius=4, num_players=2)
    # Drop a wall of dead cells along the entire q == 0 column, splitting the
    # board into a left half (q < 0) and a right half (q > 0). Hex neighbors
    # mean q == 0 cells are the only path between them, so killing them all
    # forces the two halves to disconnect.
    is_dead = np.zeros(base.N, dtype=np.bool_)
    for i in range(base.N):
        q = int(base.coord[i, 0])
        if q == 0:
            is_dead[i] = True
    dead = np.where(is_dead)[0].astype(np.int32)
    # Pick two seats, one on each side of the wall.
    left_side = [i for i in range(base.N) if base.coord[i, 0] < 0]
    right_side = [i for i in range(base.N) if base.coord[i, 0] > 0]
    seats = np.array([left_side[0], right_side[0]], dtype=np.int32)
    # Pre-carve: should be disconnected.
    assert not seats_mutually_reachable(seats, dead, base.neighbors)
    new_dead, carved = carve_seat_connectors(seats, dead, base.neighbors)
    # Post-carve: connected, and at least one cell was revived.
    assert len(carved) > 0
    assert seats_mutually_reachable(seats, new_dead, base.neighbors)
    is_new_dead = np.zeros(base.N, dtype=np.bool_)
    is_new_dead[new_dead] = True
    assert _live_subgraph_connected(is_new_dead, base.neighbors)


def test_max_seat_pair_distance_disconnected_returns_minus_one():
    """When seats are in different live components, the distance is -1
    (sentinel for unreachable)."""
    base = make_board(radius=3, num_players=2)
    is_dead = np.zeros(base.N, dtype=np.bool_)
    for k in range(base.neighbors.shape[1]):
        d = int(base.neighbors[0, k])
        if d >= 0:
            is_dead[d] = True
    dead = np.where(is_dead)[0].astype(np.int32)
    far_seat = int(np.where(~is_dead)[0][-1])
    seats = np.array([0, far_seat], dtype=np.int32)
    assert max_seat_pair_distance(seats, dead, base.neighbors) == -1


# --------------------- solvers run cleanly on connected boards ---------------------


def test_bfs_and_lightning_solvers_emit_actions_on_connected_board():
    """Smoke: with a guaranteed-connected board (retry mode), both solvers
    return action arrays of the right shape and at least some non-NOOP
    actions for each seat."""
    from flux_v2 import ACTION_NOOP
    from flux_v2.solver import solver_actions
    from flux_v2.solver_lightning import lightning_solver_actions

    rng = np.random.default_rng(123)
    state, _dead = _build_initial_state(6, 6, 10, rng, connect_mode="retry")
    for seat in range(6):
        bfs_actions = solver_actions(state, seat, rng=rng)
        lit_actions = lightning_solver_actions(state, seat, rng=rng)
        assert bfs_actions.shape == (state.N,)
        assert lit_actions.shape == (state.N,)
        # Seat owns at least one cell; that cell should produce a non-NOOP
        # because there's nothing on it yet — pure SET territory.
        owned = np.where(state.owner == seat)[0]
        assert (bfs_actions[owned] != ACTION_NOOP).any()
        assert (lit_actions[owned] != ACTION_NOOP).any()
