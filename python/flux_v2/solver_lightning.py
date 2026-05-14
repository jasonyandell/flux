"""Lightning solver for flux v2 — purely local potential-field heuristic.

No global computation, no priority queue. Each cell stores a single scalar
"potential" computed by local diffusion from intrinsic sources, then sets
outflows toward the steepest-uphill neighbor. Lightning behavior emerges:
when one enemy is much weaker than the rest, its gradient dominates the
network and pressure naturally channels toward it. When it dies, the next
attractor takes over automatically.

Per-cell intrinsic source (read fresh each AI tick):
    enemy cell        weak_bonus * (1 - strength/MAX_STRENGTH)
    neutral cell      expand_bonus  (small constant pull)
    friendly cell     defense_bonus * (inbound_enemy_pressure / MAX_EDGE)
                      (threatened back-line cells glow → relays swerve to save)
    dead cell         0  (walls don't conduct)

Diffusion (Bellman-style local update, iterated to convergence):
    pot[c] = max(intrinsic[c], gamma * max(pot[d] for d in non-dead neighbors))

Action rule (per owned cell):
    desired-on slots = those pointing at the steepest-uphill neighbor(s)
                       (pot[d] strictly greater than pot[c])
    one action this tick:  SET a missing on-slot > CLEAR a stale slot > NOOP
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .solver import _pick
from .state import (
    ACTION_CLEAR_BASE,
    ACTION_NOOP,
    ACTION_SET_BASE,
    DEAD,
    K,
    MAX_EDGE,
    MAX_STRENGTH,
    NEUTRAL,
    OPPOSITE_SLOT,
    State,
)


def _inbound_enemy_pressure(state: State, seat: int) -> np.ndarray:
    """Sum of edge_pressure on inbound edges from enemy seats. (N,) float32."""
    N = state.N
    owner = state.owner
    nb = state.neighbors
    ep = state.edge_pressure

    inbound = np.zeros(N, dtype=np.float32)
    for k in range(K):
        d_ids = nb[:, k]
        valid = d_ids >= 0
        if not valid.any():
            continue
        opp = int(OPPOSITE_SLOT[k])
        valid_d = d_ids[valid]
        press = np.zeros(N, dtype=np.float32)
        press[valid] = ep[valid_d, opp]
        d_owner = np.full(N, NEUTRAL, dtype=np.int32)
        d_owner[valid] = owner[valid_d]
        is_enemy_src = valid & (d_owner != seat) & (d_owner >= 0)
        inbound += press * is_enemy_src.astype(np.float32)
    return inbound


def compute_potential(
    state: State,
    seat: int,
    gamma: float = 0.85,
    weak_bonus: float = 1.0,
    expand_bonus: float = 0.6,
    defense_bonus: float = 0.0,
    max_iter: int = 32,
    tol: float = 1e-4,
) -> np.ndarray:
    """Return (N,) float32 potential field for `seat`. All reads are local —
    each iteration only inspects each cell's 6 neighbors."""
    N = state.N
    owner = state.owner
    strength = state.strength
    nb = state.neighbors

    intrinsic = np.zeros(N, dtype=np.float32)

    is_enemy = (owner >= 0) & (owner != seat)
    if is_enemy.any():
        intrinsic[is_enemy] = (
            weak_bonus * (1.0 - strength[is_enemy] / MAX_STRENGTH)
        ).clip(0.0, weak_bonus)

    is_neutral = owner == NEUTRAL
    intrinsic[is_neutral] = expand_bonus

    is_mine = owner == seat
    if is_mine.any():
        inbound = _inbound_enemy_pressure(state, seat)
        threat = (inbound / MAX_EDGE).clip(0.0, 1.0)
        intrinsic[is_mine] = defense_bonus * threat[is_mine]

    is_dead = owner == DEAD
    intrinsic[is_dead] = 0.0

    pot = intrinsic.copy()
    nb_safe = np.maximum(nb, 0)
    nb_valid = (nb >= 0)

    for _ in range(max_iter):
        # Gather neighbor potentials: (N, K) of pot values, zeroed where invalid.
        nbr_pot = pot[nb_safe]                                  # (N, K)
        nbr_pot = nbr_pot * nb_valid.astype(np.float32)
        # Dead neighbors don't conduct.
        nbr_dead = (owner[nb_safe] == DEAD) & nb_valid
        nbr_pot = np.where(nbr_dead, 0.0, nbr_pot)
        max_nbr = nbr_pot.max(axis=1)
        new_pot = np.maximum(intrinsic, gamma * max_nbr)
        new_pot[is_dead] = 0.0
        diff = np.abs(new_pot - pot).max()
        pot = new_pot
        if diff < tol:
            break
    return pot


def lightning_solver_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
    gamma: float = 0.85,
    weak_bonus: float = 1.0,
    expand_bonus: float = 0.6,
    defense_bonus: float = 0.0,
    fanout_eps: float = 0.05,
) -> np.ndarray:
    """Return (N,) int32 actions for `seat`. Cells not owned by `seat` get NOOP.
    Owned cells emit the one action that nudges their outflow toward the
    steepest-uphill neighbor in the local potential field."""
    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow

    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions

    pot = compute_potential(
        state, seat,
        gamma=gamma, weak_bonus=weak_bonus,
        expand_bonus=expand_bonus, defense_bonus=defense_bonus,
    )

    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        # Frontier slots: any non-friendly, non-dead neighbor — always attack.
        # This is the air-breakdown rule: pressure discharges at any exposed
        # boundary. No "is this the weakest" check; if you touch it, you
        # pump pressure into it.
        attack = np.zeros(K, dtype=np.bool_)
        relay = np.zeros(K, dtype=np.bool_)
        my_pot = pot[c]
        best_friendly_pot = my_pot
        best_friendly_slots: list[int] = []
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            if od != seat:
                attack[k] = True
                continue
            # Friendly: candidate for relay if strictly uphill.
            pd = pot[d]
            if pd > best_friendly_pot + fanout_eps:
                best_friendly_pot = pd
                best_friendly_slots = [k]
            elif abs(pd - best_friendly_pot) <= fanout_eps and pd > my_pot:
                best_friendly_slots.append(k)
        for k in best_friendly_slots:
            relay[k] = True
        desired = attack | relay

        cur = outflow[c]
        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
            continue
    return actions
