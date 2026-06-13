"""Gate 2 of the beat-the-solver plan: the Ring 1 field policy at
champion-init must reproduce `lightning_sum_throttled` action-for-action
across live game states (same pot warm-start trajectory, same picker RNG
consumption). See wiki/topics/v2-beat-the-solver-plan.md — failure here is
an implementation bug by definition.

Both policies warm-start their Bellman solve from their own previous-tick
field, so each must be called exactly once per (state, seat) — the driver
below uses the champion call that's being compared to also advance the game.

Also sanity-checks that the genome's extra degrees of freedom are live
(a perturbed genome diverges from the champion somewhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from flux_v2 import apply_actions, tick
from flux_v2.ring1 import (
    GENOME_NAMES,
    champion_vector,
    clear_field_caches,
    field_policy_actions,
)
from flux_v2.solver_lightning import lightning_solver_actions
from scripts.run_v2_solver import _build_initial_state

_IDX = {n: i for i, n in enumerate(GENOME_NAMES)}


def _champion_actions(state, seat, rng):
    return lightning_solver_actions(
        state, seat, rng=rng, mode="sum", throttle=1,
    )


def _drive_and_compare(radius, num_players, num_dead, seed, ai_steps,
                       ai_period, vector, on_diff):
    """Champion-vs-champion game. At every AI tick, every seat: run the
    champion once (its actions drive the game) and the ring1 policy once
    with an identically-seeded RNG; report mismatching cells to on_diff."""
    rng_board = np.random.default_rng(seed)
    state, _ = _build_initial_state(radius, num_players, num_dead, rng_board)
    clear_field_caches()
    for t in range(ai_steps):
        per_seat = []
        for seat in range(num_players):
            a = _champion_actions(
                state, seat, np.random.default_rng([seed, t, seat]),
            )
            b = field_policy_actions(
                state, seat, np.random.default_rng([seed, t, seat]),
                vector=vector,
            )
            if not np.array_equal(a, b):
                bad = np.where(a != b)[0]
                on_diff(t, seat, bad)
            per_seat.append(a)
        combined = per_seat[0].copy()
        for seat in range(1, num_players):
            mask = state.owner == seat
            combined[mask] = per_seat[seat][mask]
        state = apply_actions(state, combined)
        for _ in range(ai_period):
            state = tick(state)
    return state


@pytest.mark.parametrize("seed", [3, 11, 2026])
def test_ring1_champion_init_matches_champion(seed):
    mismatches = []
    _drive_and_compare(
        radius=5, num_players=4, num_dead=8, seed=seed,
        ai_steps=60, ai_period=5, vector=champion_vector(),
        on_diff=lambda t, s, bad: mismatches.append((t, s, bad[:5].tolist())),
    )
    assert not mismatches, f"ring1 diverged from champion at {mismatches[:3]}"


def test_ring1_perturbed_genome_diverges():
    """The extra degrees of freedom must be live, not dead weights."""
    vec = champion_vector()
    vec[_IDX["thr_base"]] = 2.0          # allow 2 outflows per cell
    vec[_IDX["w_defense"]] = 1.5         # defense-aware intrinsic
    diverged = []
    _drive_and_compare(
        radius=5, num_players=4, num_dead=8, seed=7,
        ai_steps=40, ai_period=5, vector=vec,
        on_diff=lambda t, s, bad: diverged.append((t, s)),
    )
    assert diverged, "perturbed genome never diverged — knobs are dead"
