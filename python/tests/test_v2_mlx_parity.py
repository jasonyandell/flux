"""Parity test: MLX batched step ≡ pure reducer.

Build a randomized v2 board, take a sequence of identical ticks, and verify
the MLX path produces the same owners + strengths + edge_pressure + outflow
as the pure reducer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mlx.core as mx

from flux_v2 import (
    ACTION_NOOP,
    ACTION_SET_BASE,
    K,
    apply_actions,
    make_board,
    tick,
    transit_credit_per_cell_for_tick,
    waste_per_cell_for_tick,
)
from flux_v2.mlx_step import apply_actions_batched, tick_batched


def _state_to_mlx(s):
    return (
        mx.array(s.owner.reshape(1, -1)),
        mx.array(s.strength.reshape(1, -1)),
        mx.array(s.outflow.reshape(1, s.N, K)),
        mx.array(s.edge_pressure.reshape(1, s.N, K)),
    )


def test_step_parity_random():
    rng = np.random.default_rng(7)
    base = make_board(radius=4, num_players=4)
    s = base
    # Random initial actions: each owned cell sets a random valid slot.
    for _ in range(3):
        actions = np.full(s.N, ACTION_NOOP, dtype=np.int32)
        for c in range(s.N):
            if s.owner[c] >= 0:
                valid_slots = [k for k in range(K) if s.neighbors[c, k] >= 0]
                if valid_slots and rng.random() < 0.5:
                    actions[c] = ACTION_SET_BASE + rng.choice(valid_slots)
        s = apply_actions(s, actions)

    # Mirror to MLX state.
    o_mx, st_mx, of_mx, ep_mx = _state_to_mlx(s)
    nb_mx = mx.array(s.neighbors)
    alive = mx.array(np.array([True]))

    # Pure and MLX paths: run 25 ticks, checking reducer state and the
    # diagnostic reward channels emitted for each tick.
    cur = s
    o = o_mx; st = st_mx; of = of_mx; ep = ep_mx
    for _ in range(25):
        waste_expected = waste_per_cell_for_tick(cur)
        transit_expected = transit_credit_per_cell_for_tick(cur)
        o, st, of, ep, waste_mx, transit_mx = tick_batched(
            o, st, of, ep, alive, s.num_players, nb_mx,
        )
        cur = tick(cur)
        mx.eval(o, st, of, ep, waste_mx, transit_mx)

        assert np.allclose(np.array(waste_mx)[0], waste_expected, atol=1e-4)
        assert np.allclose(np.array(transit_mx)[0], transit_expected, atol=1e-4)

    # Compare.
    o_np = np.array(o)[0]
    st_np = np.array(st)[0]
    of_np = np.array(of)[0]
    ep_np = np.array(ep)[0]

    assert np.array_equal(o_np, cur.owner), \
        f"owner mismatch: max diff cells {np.where(o_np != cur.owner)[0][:10].tolist()}"
    assert np.allclose(st_np, cur.strength, atol=1e-3), \
        f"strength max diff {np.abs(st_np - cur.strength).max()}"
    assert np.array_equal(of_np, cur.outflow), "outflow mismatch"
    assert np.allclose(ep_np, cur.edge_pressure, atol=1e-3), \
        f"edge_pressure max diff {np.abs(ep_np - cur.edge_pressure).max()}"


def test_action_apply_parity():
    rng = np.random.default_rng(1)
    s = make_board(radius=4, num_players=4)
    actions = np.full(s.N, ACTION_NOOP, dtype=np.int32)
    for c in range(s.N):
        if s.owner[c] >= 0 and rng.random() < 0.5:
            valid = [k for k in range(K) if s.neighbors[c, k] >= 0]
            if valid:
                actions[c] = ACTION_SET_BASE + rng.choice(valid)

    s_py = apply_actions(s, actions)

    o_mx, _, of_mx, _ = _state_to_mlx(s)
    nb_mx = mx.array(s.neighbors)
    new_of = apply_actions_batched(
        o_mx, of_mx, mx.array(actions.reshape(1, -1)), nb_mx,
    )
    mx.eval(new_of)
    assert np.array_equal(np.array(new_of)[0], s_py.outflow), "action apply parity"
