"""Ring 2 parity: the opponent-field extension must reduce EXACTLY to Ring 1
(hence to the champion `lightning_sum_throttled`) when the three opponent
weights are zero — that's the champion-init guarantee that lets ES start from
a known-good genome. See the Ring 1 parity test (same driver structure).

`field_policy2_actions` at `champion2_vector()` (= champion ++ [0,0,0]) skips
the opponent field entirely and must match the champion action-for-action
across live game states. A perturbed Ring 2 genome (atk_opp=1.5) must turn the
opponent field live and diverge somewhere — proof the new knobs aren't dead.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from flux_v2 import apply_actions, tick
from flux_v2.ring1 import (
    GENOME2_NAMES,
    champion2_vector,
    clear_field_caches,
    field_policy2_actions,
)
from flux_v2.solver_lightning import lightning_solver_actions
from scripts.run_v2_solver import _build_initial_state

_IDX2 = {n: i for i, n in enumerate(GENOME2_NAMES)}


def _champion_actions(state, seat, rng):
    return lightning_solver_actions(
        state, seat, rng=rng, mode="sum", throttle=1,
    )


def _drive_and_compare(radius, num_players, num_dead, seed, ai_steps,
                       ai_period, vector, on_diff):
    """Champion-vs-champion game. At every AI tick, every seat: run the
    champion once (its actions drive the game) and the ring2 policy once
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
            b = field_policy2_actions(
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
def test_ring2_champion_init_matches_champion(seed):
    mismatches = []
    _drive_and_compare(
        radius=5, num_players=4, num_dead=8, seed=seed,
        ai_steps=60, ai_period=5, vector=champion2_vector(),
        on_diff=lambda t, s, bad: mismatches.append((t, s, bad[:5].tolist())),
    )
    assert not mismatches, f"ring2 diverged from champion at {mismatches[:3]}"


def test_ring2_perturbed_genome_diverges():
    """The opponent-field knobs must be live: turning on atk_opp must change
    some action.

    NOTE: a *positive* atk_opp (the spec's example value 1.5) is provably a
    no-op at champion-init — the champion attack gate is `atk_bias=1.0` with
    `atk_pot=atk_weak=0`, so it's already > 0 for every attackable slot, and
    adding a non-negative `atk_opp*opp_pot[d]` (opp_pot in [0,1]) only pushes
    it further positive: it can never flip an attack off. We therefore use
    atk_opp=-1.5 — same knob, "avoid contested targets" per the spec — which
    suppresses attacks on cells the enemy bloc covets and genuinely diverges.
    """
    vec = champion2_vector()
    vec[_IDX2["atk_opp"]] = -1.5
    diverged = []
    _drive_and_compare(
        radius=5, num_players=4, num_dead=8, seed=7,
        ai_steps=40, ai_period=5, vector=vec,
        on_diff=lambda t, s, bad: diverged.append((t, s)),
    )
    assert diverged, "perturbed ring2 genome never diverged — opp knobs are dead"


def test_ring2_opp_field_is_nontrivial():
    """The opponent field itself must carry signal (not all-zero), otherwise
    the opp knobs would be vacuously dead regardless of weight."""
    from flux_v2.ring1 import opp_potential
    rng_board = np.random.default_rng(7)
    state, _ = _build_initial_state(5, 4, 8, rng_board)
    clear_field_caches()
    pot = opp_potential(state, seat=0, gamma=0.85)
    assert float(pot.max()) > 0.0, "opponent field is degenerate (all zero)"


def test_ring2_rank_opp_diverges():
    """Second live opp knob: rank_opp re-orders the per-cell throttle. With a
    larger throttle budget so multiple slots compete, it must change actions."""
    vec = champion2_vector()
    vec[_IDX2["rank_opp"]] = 2.0
    diverged = []
    _drive_and_compare(
        radius=5, num_players=4, num_dead=8, seed=7,
        ai_steps=40, ai_period=5, vector=vec,
        on_diff=lambda t, s, bad: diverged.append((t, s)),
    )
    assert diverged, "rank_opp never diverged — throttle re-ranking is dead"
