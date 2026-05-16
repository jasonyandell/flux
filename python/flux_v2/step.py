"""flux v2 pure reducer.

Two entry points:

  apply_actions(state, actions) → state'     (called once per AI tick)
  tick(state) → state'                       (called once per game tick)

Both are pure: no in-place mutation of the input. The reducer fully implements
the per-tick algorithm and mutation invariants from the v2 PRD.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .state import (
    ACTION_CLEAR_BASE,
    ACTION_NOOP,
    ACTION_SET_BASE,
    CAPTURE_STRENGTH,
    DEAD,
    EDGE_ALPHA,
    K,
    MAX_EDGE,
    MAX_STRENGTH,
    NEUTRAL,
    OPPOSITE_SLOT,
    REGEN_BASE_PER_TICK,
    State,
    TRANSIT_CREDIT_STRICT,
    WASTE_WEIGHT_CAP_BOUND,
    WASTE_WEIGHT_DEST_TERMINATED,
    WASTE_WEIGHT_NO_SPILL,
    copy_state,
    regen,
)


# Module-level int32 OPPOSITE_SLOT view; passed into @njit cores.
_OPPOSITE_SLOT_ARR = np.asarray(OPPOSITE_SLOT, dtype=np.int32)


# ---------------------------------------------------------------------------
# AI-tick action application
# ---------------------------------------------------------------------------

@njit(cache=True)
def _apply_actions_core(
    owner: np.ndarray,
    outflow: np.ndarray,
    actions: np.ndarray,
    neighbors: np.ndarray,
    opposite_slot: np.ndarray,
    action_set_base: int,
    action_clear_base: int,
    action_noop: int,
    dead_id: int,
) -> np.ndarray:
    """Apply one action per cell with friendly-bidirectional resolution.

    Returns new_outflow (N, K) bool. Pure: input outflow not mutated.
    """
    N = owner.shape[0]
    K = neighbors.shape[1]
    new_outflow = outflow.copy()
    # Track newly-set friendly slots for bidirectional resolution.
    # fresh_friendly[c, k] = True if this action set c→d (d friendly) anew.
    fresh_friendly = np.zeros((N, K), dtype=np.bool_)

    for c in range(N):
        a = actions[c]
        oc = owner[c]
        if oc < 0:
            continue
        if a == action_noop:
            continue
        if a >= action_set_base and a < action_clear_base:
            k = a - action_set_base
            d = neighbors[c, k]
            if d < 0:
                continue
            if owner[d] == dead_id:
                continue
            if new_outflow[c, k]:
                continue
            new_outflow[c, k] = True
            if owner[d] == oc:
                fresh_friendly[c, k] = True
        elif a >= action_clear_base and a < action_noop:
            k = a - action_clear_base
            d = neighbors[c, k]
            if d < 0:
                continue
            new_outflow[c, k] = False

    # Resolve friendly bidirectional: for each fresh c→d, check back edge.
    # Higher cell-index wins; loser's slot is cleared.
    for c in range(N):
        for k in range(K):
            if not fresh_friendly[c, k]:
                continue
            d = neighbors[c, k]
            opp_k = opposite_slot[k]
            if not new_outflow[d, opp_k]:
                continue
            if c > d:
                new_outflow[d, opp_k] = False
            else:
                new_outflow[c, k] = False
                fresh_friendly[c, k] = False
    return new_outflow


def apply_actions(state: State, actions: np.ndarray) -> State:
    """Apply one action per cell. Pure: returns a fresh State.

    Invariants:
      1. No friendly bidirectional flow — for each freshly-set c→d with d
         friendly, if the back-edge d→c is also set, lower cell-index side
         is cleared (higher wins).
      2. Capture-clears handled in `tick`, not here.
      3. Stale targets stay on (only owned-cell actions apply).
    """
    s = copy_state(state)
    if actions.shape != (s.N,):
        raise ValueError(f"actions shape mismatch: {actions.shape} vs ({s.N},)")
    s.outflow = _apply_actions_core(
        np.ascontiguousarray(s.owner, dtype=np.int32),
        np.ascontiguousarray(s.outflow, dtype=np.bool_),
        np.ascontiguousarray(actions, dtype=np.int32),
        np.ascontiguousarray(s.neighbors, dtype=np.int32),
        _OPPOSITE_SLOT_ARR,
        int(ACTION_SET_BASE),
        int(ACTION_CLEAR_BASE),
        int(ACTION_NOOP),
        int(DEAD),
    )
    return s


# ---------------------------------------------------------------------------
# Per-tick physics
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _tick_core(
    owner: np.ndarray,
    strength: np.ndarray,
    outflow: np.ndarray,
    edge_pressure_prev: np.ndarray,
    neighbors: np.ndarray,
    opposite_slot: np.ndarray,
    num_players: int,
    edge_alpha: float,
    max_strength: float,
    max_edge: float,
    regen_base: float,
    w_no_spill: float,
    w_cap_bound: float,
    w_dest_terminated: float,
    dead_id: int,
    neutral_id: int,
):
    """JIT'd per-tick physics. Pure: returns fresh arrays.

    Mirrors the numpy version semantically; structured as explicit per-
    cell loops because Numba optimizes loop-and-arithmetic better than
    chains of (N, K) numpy expressions, and small-array dispatch
    overhead per numpy call dominates pure-Python at our board sizes.
    """
    N = owner.shape[0]
    K = neighbors.shape[1]

    new_owner = owner.copy()
    new_strength = strength.copy()
    new_outflow = outflow.copy()
    new_edge_pressure = np.zeros((N, K), dtype=np.float32)

    overflow_arr = np.zeros(N, dtype=np.float32)
    is_alive_arr = np.zeros(N, dtype=np.bool_)
    enemy_pressure_by_player = np.zeros((N, num_players), dtype=np.float32)
    num_active = np.zeros(N, dtype=np.int32)
    per_edge = np.zeros(N, dtype=np.float32)
    can_spill_arr = np.zeros(N, dtype=np.bool_)

    waste_delta = 0.0

    # Pass 1: gather inbound pressure, apply fill, detect capture.
    for c in range(N):
        oc = owner[c]
        sc = strength[c]
        is_dead_c = oc == dead_id
        is_alive_c = oc >= 0
        is_neutral_c = oc == neutral_id
        is_alive_arr[c] = is_alive_c

        if is_dead_c:
            new_owner[c] = dead_id
            new_strength[c] = 0.0
            continue

        p_in_friendly = regen_base if is_alive_c else 0.0
        p_in_enemy = 0.0
        for k in range(K):
            d = neighbors[c, k]
            if d < 0:
                continue
            press = edge_pressure_prev[d, opposite_slot[k]]
            od = owner[d]
            if od == dead_id:
                continue
            if od == oc and is_alive_c:
                p_in_friendly += press
            elif od != oc and od >= 0:
                p_in_enemy += press
                enemy_pressure_by_player[c, od] += press

        if sc < max_strength:
            headroom = max_strength - sc
        else:
            headroom = 0.0
        grew = p_in_friendly if p_in_friendly < headroom else headroom
        of = (p_in_friendly - grew) if is_alive_c else 0.0
        pre_strength = sc + grew - p_in_enemy
        overflow_arr[c] = of

        will_capture = False
        cap_pre_strength = pre_strength
        if is_alive_c and pre_strength < 0.0 and p_in_enemy > 0.0:
            will_capture = True
        elif is_neutral_c:
            neutral_pre = sc - p_in_enemy
            if neutral_pre < 0.0 and p_in_enemy > 0.0:
                will_capture = True
                cap_pre_strength = neutral_pre

        if will_capture:
            best_p = 0
            best_pressure = enemy_pressure_by_player[c, 0]
            for p in range(1, num_players):
                ep = enemy_pressure_by_player[c, p]
                if ep > best_pressure:
                    best_pressure = ep
                    best_p = p
            new_owner[c] = best_p
            surplus = -cap_pre_strength
            if surplus < 0.0:
                surplus = 0.0
            if surplus > max_strength:
                surplus = max_strength
            new_strength[c] = surplus
            for k in range(K):
                new_outflow[c, k] = False
        else:
            if is_alive_c:
                ns = pre_strength
                if ns < 0.0:
                    ns = 0.0
                if ns > max_strength:
                    ns = max_strength
                new_strength[c] = ns
            elif is_neutral_c:
                ns = sc - p_in_enemy
                if ns < 0.0:
                    ns = 0.0
                if ns > max_strength:
                    ns = max_strength
                new_strength[c] = ns
            else:
                new_strength[c] = sc

    # Pass 2: num_active on post-capture outflow.
    for c in range(N):
        n = 0
        for k in range(K):
            if new_outflow[c, k]:
                n += 1
        num_active[c] = n

    # Pass 3: per_edge + waste (cap-bound and no-spill).
    for c in range(N):
        if not is_alive_arr[c]:
            continue
        of = overflow_arr[c]
        if of <= 0.0:
            continue
        na = num_active[c]
        if na > 0:
            pe = of / na
            pe_clipped = pe if pe < max_edge else max_edge
            per_edge[c] = pe_clipped
            can_spill_arr[c] = True
            waste_delta += w_cap_bound * (pe - pe_clipped) * na
        else:
            waste_delta += w_no_spill * of

    # Pass 4: target_edge_pressure + momentum blend.
    if edge_alpha >= 1.0:
        for c in range(N):
            cs = can_spill_arr[c]
            for k in range(K):
                if cs and new_outflow[c, k]:
                    new_edge_pressure[c, k] = per_edge[c]
                else:
                    new_edge_pressure[c, k] = 0.0
    else:
        one_minus_alpha = 1.0 - edge_alpha
        for c in range(N):
            cs = can_spill_arr[c]
            for k in range(K):
                target = per_edge[c] if (cs and new_outflow[c, k]) else 0.0
                new_edge_pressure[c, k] = (
                    one_minus_alpha * edge_pressure_prev[c, k]
                    + edge_alpha * target
                )

    # Pass 5: dest-terminated waste.
    if w_dest_terminated > 0.0:
        for c in range(N):
            if not can_spill_arr[c]:
                continue
            oc = owner[c]
            for k in range(K):
                if not new_outflow[c, k]:
                    continue
                d = neighbors[c, k]
                if d < 0:
                    continue
                if (
                    owner[d] == oc
                    and strength[d] >= max_strength
                    and num_active[d] == 0
                ):
                    waste_delta += w_dest_terminated * per_edge[c]

    return new_owner, new_strength, new_outflow, new_edge_pressure, waste_delta


def tick(state: State, edge_alpha: float | None = None) -> State:
    """Apply the per-tick algorithm. Returns a fresh state.

    edge_alpha controls edge-pressure momentum:
      next = (1 - alpha) * old + alpha * target
    alpha=1.0 → snap to target (original v2 physics, default).
    alpha<1.0 → fluid-style buildup. None falls back to the module-level
    EDGE_ALPHA constant.
    """
    if edge_alpha is None:
        edge_alpha = EDGE_ALPHA
    s = copy_state(state)
    new_owner, new_strength, new_outflow, new_edge_pressure, waste_delta = _tick_core(
        np.ascontiguousarray(s.owner, dtype=np.int32),
        np.ascontiguousarray(s.strength, dtype=np.float32),
        np.ascontiguousarray(s.outflow, dtype=np.bool_),
        np.ascontiguousarray(s.edge_pressure, dtype=np.float32),
        np.ascontiguousarray(s.neighbors, dtype=np.int32),
        _OPPOSITE_SLOT_ARR,
        max(s.num_players, 1),
        float(edge_alpha),
        float(MAX_STRENGTH),
        float(MAX_EDGE),
        float(REGEN_BASE_PER_TICK),
        float(WASTE_WEIGHT_NO_SPILL),
        float(WASTE_WEIGHT_CAP_BOUND),
        float(WASTE_WEIGHT_DEST_TERMINATED),
        int(DEAD),
        int(NEUTRAL),
    )
    s.owner = new_owner
    s.strength = new_strength
    s.outflow = new_outflow
    s.edge_pressure = new_edge_pressure
    s.tick = state.tick + 1
    s.waste_total = state.waste_total + float(waste_delta)
    return s


def waste_per_cell_for_tick(state: State) -> np.ndarray:
    """Diagnostic: compute the per-cell waste that *would be* attributed
    if `tick(state)` were applied right now. Useful for per-seat reward shaping.

    Returns (N,) float32. Sums the two waste sources (cap-bound + no-outflow).
    """
    N = state.N
    owner = state.owner
    strength = state.strength
    nb = state.neighbors
    outflow = state.outflow
    edge_pressure_prev = state.edge_pressure
    is_alive = owner >= 0

    pressure_in_friendly = np.zeros(N, dtype=np.float32)
    pressure_in_friendly[is_alive] = regen(strength[is_alive]).astype(np.float32)
    for k in range(K):
        d_ids = nb[:, k]
        valid = d_ids >= 0
        opp = int(OPPOSITE_SLOT[k])
        pressure = np.zeros(N, dtype=np.float32)
        valid_d = d_ids[valid]
        pressure[valid] = edge_pressure_prev[valid_d, opp]
        d_owner = np.full(N, NEUTRAL, dtype=np.int32)
        d_owner[valid] = owner[valid_d]
        is_friendly_d = valid & (d_owner == owner) & is_alive
        pressure_in_friendly += pressure * is_friendly_d.astype(np.float32)

    headroom = np.maximum(MAX_STRENGTH - strength, 0.0)
    grew = np.minimum(pressure_in_friendly, headroom)
    overflow = pressure_in_friendly - grew
    overflow = np.where(is_alive, overflow, 0.0)
    num_active = outflow.sum(axis=1).astype(np.int32)
    can_spill = (num_active > 0) & (overflow > 0) & is_alive
    no_spill = is_alive & (overflow > 0) & (num_active == 0)

    waste = np.zeros(N, dtype=np.float32)
    per_edge = np.zeros(N, dtype=np.float32)
    if can_spill.any():
        per_edge_unclipped = np.zeros(N, dtype=np.float32)
        per_edge_unclipped[can_spill] = overflow[can_spill] / num_active[can_spill]
        per_edge = np.minimum(per_edge_unclipped, MAX_EDGE)
        excess_per_slot = (per_edge_unclipped - per_edge)
        waste[can_spill] += (
            WASTE_WEIGHT_CAP_BOUND * excess_per_slot[can_spill] * num_active[can_spill]
        )
    if no_spill.any():
        waste[no_spill] += WASTE_WEIGHT_NO_SPILL * overflow[no_spill]
    if WASTE_WEIGHT_DEST_TERMINATED > 0.0 and can_spill.any():
        for k in range(K):
            d_ids = nb[:, k]
            slot_active = outflow[:, k] & (d_ids >= 0) & can_spill
            if not slot_active.any():
                continue
            src_idx = np.where(slot_active)[0]
            d_idx = d_ids[src_idx]
            is_dead_end = (
                (owner[d_idx] == owner[src_idx])
                & (strength[d_idx] >= MAX_STRENGTH)
                & (num_active[d_idx] == 0)
            )
            if is_dead_end.any():
                terminated_src = src_idx[is_dead_end]
                waste[terminated_src] += (
                    WASTE_WEIGHT_DEST_TERMINATED * per_edge[terminated_src]
                )
    return waste


def transit_credit_per_cell_for_tick(state: State) -> np.ndarray:
    """Diagnostic: compute per-source transit credit for the next tick.

    A source earns credit when its active outflow carries overflow pressure
    into a friendly relay: a destination with active outflows of its own. In
    strict mode, the destination must also already be at MAX strength, which
    makes this the positive mirror of dest-terminated waste without rewarding
    ordinary fill traffic.
    """
    N = state.N
    owner = state.owner
    strength = state.strength
    nb = state.neighbors
    outflow = state.outflow
    edge_pressure_prev = state.edge_pressure
    is_alive = owner >= 0

    pressure_in_friendly = np.zeros(N, dtype=np.float32)
    pressure_in_friendly[is_alive] = regen(strength[is_alive]).astype(np.float32)
    for k in range(K):
        d_ids = nb[:, k]
        valid = d_ids >= 0
        opp = int(OPPOSITE_SLOT[k])
        pressure = np.zeros(N, dtype=np.float32)
        valid_d = d_ids[valid]
        pressure[valid] = edge_pressure_prev[valid_d, opp]
        d_owner = np.full(N, NEUTRAL, dtype=np.int32)
        d_owner[valid] = owner[valid_d]
        is_friendly_d = valid & (d_owner == owner) & is_alive
        pressure_in_friendly += pressure * is_friendly_d.astype(np.float32)

    headroom = np.maximum(MAX_STRENGTH - strength, 0.0)
    grew = np.minimum(pressure_in_friendly, headroom)
    overflow = pressure_in_friendly - grew
    overflow = np.where(is_alive, overflow, 0.0)
    num_active = outflow.sum(axis=1).astype(np.int32)
    can_spill = (num_active > 0) & (overflow > 0) & is_alive

    credit = np.zeros(N, dtype=np.float32)
    if not can_spill.any():
        return credit

    per_edge_unclipped = np.zeros(N, dtype=np.float32)
    per_edge_unclipped[can_spill] = overflow[can_spill] / num_active[can_spill]
    per_edge = np.minimum(per_edge_unclipped, MAX_EDGE)

    for k in range(K):
        d_ids = nb[:, k]
        slot_active = outflow[:, k] & (d_ids >= 0) & can_spill
        if not slot_active.any():
            continue
        src_idx = np.where(slot_active)[0]
        d_idx = d_ids[src_idx]
        is_relay = (
            (owner[d_idx] == owner[src_idx])
            & (num_active[d_idx] > 0)
        )
        if TRANSIT_CREDIT_STRICT:
            is_relay &= strength[d_idx] >= MAX_STRENGTH
        if is_relay.any():
            relay_src = src_idx[is_relay]
            credit[relay_src] += per_edge[relay_src]

    return credit
