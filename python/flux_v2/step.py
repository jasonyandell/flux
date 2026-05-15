"""flux v2 pure reducer.

Two entry points:

  apply_actions(state, actions) → state'     (called once per AI tick)
  tick(state) → state'                       (called once per game tick)

Both are pure: no in-place mutation of the input. The reducer fully implements
the per-tick algorithm and mutation invariants from the v2 PRD.
"""
from __future__ import annotations

import numpy as np

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
    State,
    TRANSIT_CREDIT_STRICT,
    WASTE_WEIGHT_CAP_BOUND,
    WASTE_WEIGHT_DEST_TERMINATED,
    WASTE_WEIGHT_NO_SPILL,
    copy_state,
    regen,
)


# ---------------------------------------------------------------------------
# AI-tick action application
# ---------------------------------------------------------------------------

def apply_actions(state: State, actions: np.ndarray) -> State:
    """Apply one action per cell. Fully vectorized over (N, K).

    Action 0..K-1 = set slot k; K..2K-1 = clear slot k-K; 2K = no-op.
    Actions on unowned cells are masked out.

    Invariants:
      1. No friendly bidirectional flow — for each freshly-set c→d with d
         friendly, if the back-edge d→c is also set, the lower cell-index
         side is cleared. Higher cell-index wins ties.
      2. Capture-clears handled in `tick`, not here.
      3. Stale targets stay on (only owned-cell actions apply).
    """
    s = copy_state(state)
    N = s.N
    owner = s.owner
    nb = s.neighbors
    if actions.shape != (N,):
        raise ValueError(f"actions shape mismatch: {actions.shape} vs ({N},)")

    is_owned = owner >= 0
    nb_safe = np.maximum(nb, 0)
    nb_valid = nb >= 0
    nb_owner = np.where(nb_valid, owner[nb_safe], NEUTRAL)
    nb_is_dead = nb_valid & (nb_owner == DEAD)

    slot_idx = np.arange(K, dtype=np.int32)
    set_mask_cell = (actions >= ACTION_SET_BASE) & (actions < ACTION_CLEAR_BASE)
    clear_mask_cell = (actions >= ACTION_CLEAR_BASE) & (actions < ACTION_NOOP)
    set_slot = np.where(set_mask_cell, actions, 0)
    clear_slot = np.where(clear_mask_cell, actions - ACTION_CLEAR_BASE, 0)
    set_per_slot = (
        set_mask_cell[:, None]
        & (set_slot[:, None] == slot_idx[None, :])
        & is_owned[:, None]
        & nb_valid
        & ~nb_is_dead
    )
    clear_per_slot = (
        clear_mask_cell[:, None]
        & (clear_slot[:, None] == slot_idx[None, :])
        & is_owned[:, None]
        & nb_valid
    )

    new_outflow = s.outflow.copy()
    fresh_set = set_per_slot & ~new_outflow          # only newly-on triggers invariant
    new_outflow = new_outflow | fresh_set
    new_outflow = new_outflow & ~clear_per_slot

    # Bidirectional friendly: check back-edge on fresh friendly SETs.
    fresh_friendly = fresh_set & (nb_owner == owner[:, None]) & is_owned[:, None]
    if fresh_friendly.any():
        opp = OPPOSITE_SLOT
        slot_grid = np.broadcast_to(opp[None, :], (N, K))
        back_set = new_outflow[nb_safe, slot_grid]    # outflow[d, opp(k)]
        conflict = fresh_friendly & back_set
        if conflict.any():
            cell_idx_grid = np.broadcast_to(
                np.arange(N, dtype=np.int32)[:, None], (N, K),
            )
            higher_wins_c = cell_idx_grid > nb_safe   # c > d → c wins (clear d)
            clear_back = conflict & higher_wins_c
            clear_self = conflict & ~higher_wins_c
            if clear_back.any():
                cs, ks = np.where(clear_back)
                ds = nb_safe[cs, ks]
                opps = opp[ks]
                new_outflow[ds, opps] = False
            if clear_self.any():
                cs, ks = np.where(clear_self)
                new_outflow[cs, ks] = False

    s.outflow = new_outflow
    return s


# ---------------------------------------------------------------------------
# Per-tick physics
# ---------------------------------------------------------------------------

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
    N = s.N
    owner = s.owner
    strength = s.strength
    nb = s.neighbors
    outflow = s.outflow
    edge_pressure_prev = s.edge_pressure   # read end-of-last-tick

    # Per-cell "alive" mask (owned & not dead).
    is_alive = owner >= 0
    is_dead = owner == DEAD
    # Pressure can land on neutrals too (to capture them). DEAD walls absorb
    # nothing. Gate inbound accumulation on `not_dead`, not on `is_alive` —
    # the latter excludes neutral receivers and breaks neutral capture.
    not_dead = ~is_dead

    # ---- Gather pressure_in_friendly and pressure_in_enemy at each cell ----
    # Vectorized over the K-loop: build (N, K) grids of pressure, neighbor
    # owner, and friendly/enemy masks, then collapse along axis 1.
    enemy_pressure_by_player = np.zeros((N, max(s.num_players, 1)), dtype=np.float32)

    pressure_in_friendly = np.zeros(N, dtype=np.float32)
    pressure_in_friendly[is_alive] = regen(strength[is_alive]).astype(np.float32)

    nb_safe = np.maximum(nb, 0)
    nb_valid = nb >= 0
    opp = OPPOSITE_SLOT                                       # (K,)
    # Pressure flowing in: edge_pressure_prev[d, opp(k)] for d = nb[c, k].
    pressure_grid = edge_pressure_prev[nb_safe, opp[None, :]] # (N, K)
    pressure_grid = np.where(nb_valid, pressure_grid, 0.0)
    # Neighbor owner per (c, k).
    nb_owner = np.where(nb_valid, owner[nb_safe], NEUTRAL)    # (N, K)
    is_friendly_d = nb_valid & (nb_owner == owner[:, None]) & is_alive[:, None]
    is_enemy_d = (
        nb_valid
        & (nb_owner != owner[:, None])
        & (nb_owner >= 0)
        & not_dead[:, None]
    )
    pressure_in_friendly = pressure_in_friendly + (
        pressure_grid * is_friendly_d.astype(np.float32)
    ).sum(axis=1)
    pressure_in_enemy = (
        pressure_grid * is_enemy_d.astype(np.float32)
    ).sum(axis=1)
    # Attribute enemy pressure by the source player (for capture resolution).
    if is_enemy_d.any():
        c_idx, k_idx = np.where(is_enemy_d)
        attackers = nb_owner[c_idx, k_idx]
        np.add.at(
            enemy_pressure_by_player,
            (c_idx, attackers),
            pressure_grid[c_idx, k_idx],
        )

    # ---- Fill first ----
    headroom = np.maximum(MAX_STRENGTH - strength, 0.0)
    grew = np.minimum(pressure_in_friendly, headroom)
    overflow = pressure_in_friendly - grew              # excess friendly pressure
    overflow = np.where(is_alive, overflow, 0.0)
    pre_strength = strength + grew - pressure_in_enemy

    # ---- Capture detection ----
    # pre_strength < 0 with at least one enemy attacker → captured by the
    # strongest attacker.
    will_capture = (pre_strength < 0) & is_alive & (pressure_in_enemy > 0)
    # Neutral cells: capturable directly if any enemy pressure exceeded their
    # strength + grew (regen). pre_strength < 0 covers it the same way, but
    # neutrals start with is_alive = False and don't regen. Re-include them.
    is_neutral = owner == NEUTRAL
    neutral_capture_pre = strength - pressure_in_enemy
    will_capture_neutral = (neutral_capture_pre < 0) & is_neutral & (pressure_in_enemy > 0)
    capture_mask = will_capture | will_capture_neutral

    new_owner = owner.copy()
    new_strength = np.where(
        is_alive | is_neutral,
        np.clip(pre_strength, 0.0, MAX_STRENGTH),
        strength,
    ).astype(np.float32)
    new_strength = np.where(
        is_neutral,
        np.clip(neutral_capture_pre, 0.0, MAX_STRENGTH),
        new_strength,
    ).astype(np.float32)

    if capture_mask.any():
        cap_cells = np.where(capture_mask)[0]
        attackers = enemy_pressure_by_player[cap_cells].argmax(axis=1)
        new_owner[cap_cells] = attackers.astype(new_owner.dtype)
        # New: captured cell takes |pre_strength| as starting strength — i.e.
        # the surplus pressure that broke the defender. pre_strength = strength
        # + grew - pressure_in_enemy (for owned) or strength - pressure_in_enemy
        # (for neutral, grew=0). Negative when captured; surplus = -pre_strength.
        surplus = np.clip(-pre_strength[cap_cells], 0.0, MAX_STRENGTH)
        new_strength[cap_cells] = surplus

    # Dead cells stay dead.
    new_owner[is_dead] = DEAD
    new_strength[is_dead] = 0.0

    # ---- Spill (rewrite edge_pressure for next tick) ----
    # target_edge_pressure is what the edge *would* carry if it snapped
    # instantly to the source's spill share this tick (the original v2
    # behavior). new_edge_pressure is the relaxed value, blending the
    # previous tick's pressure toward that target by `edge_alpha`.
    target_edge_pressure = np.zeros_like(edge_pressure_prev)
    waste_this_tick = 0.0

    # Block 1: capturing cells lose their outflows.
    new_outflow = outflow.copy()
    if capture_mask.any():
        new_outflow[capture_mask] = False

    num_active = new_outflow.sum(axis=1).astype(np.int32)              # (N,)
    can_spill = (num_active > 0) & (overflow > 0) & is_alive
    per_edge = np.zeros(N, dtype=np.float32)

    if can_spill.any():
        per_edge_unclipped = np.zeros(N, dtype=np.float32)
        per_edge_unclipped[can_spill] = overflow[can_spill] / num_active[can_spill]
        per_edge = np.minimum(per_edge_unclipped, MAX_EDGE)
        target_edge_pressure = new_outflow * per_edge[:, None]
        # Cap-bound waste: amount per slot above MAX_EDGE (busy waste).
        excess_per_slot = (per_edge_unclipped - per_edge)             # (N,)
        waste_this_tick += WASTE_WEIGHT_CAP_BOUND * float(
            (excess_per_slot[can_spill] * num_active[can_spill]).sum()
        )

    # Relax edges toward the spill target. With alpha=1.0 this collapses
    # to the original "snap to target" behavior bit-for-bit. With alpha<1
    # edges low-pass-filter the target — pressure builds up and bleeds
    # off over ~1/alpha ticks, giving the system momentum (the "fluid"
    # framing). Capturing cells' outflows have already been cleared, so
    # their previous outflow rows decay toward 0 even though the source
    # is gone.
    if edge_alpha >= 1.0:
        new_edge_pressure = target_edge_pressure
    else:
        new_edge_pressure = (
            (1.0 - edge_alpha) * edge_pressure_prev
            + edge_alpha * target_edge_pressure
        )

    # Cells with overflow > 0 but no active outflows → all overflow is waste.
    # Weighted much heavier than cap-bound waste — a cell maxed with no
    # outflows is producing pressure that goes nowhere, the worst kind.
    no_spill = is_alive & (overflow > 0) & (num_active == 0)
    if no_spill.any():
        waste_this_tick += WASTE_WEIGHT_NO_SPILL * float(overflow[no_spill].sum())

    # Destination-terminated waste: source's active outflow points at a
    # friendly cell that is at MAX strength AND has zero active outflows.
    # Pressure lands on a pure sink — no relay, no headroom. Attribute to
    # source so the policy learns to clear those outflows. Pass-through MAX
    # cells (outflows > 0) are combo relays and stay un-penalized.
    if WASTE_WEIGHT_DEST_TERMINATED > 0.0 and can_spill.any():
        for k in range(K):
            d_ids = nb[:, k]
            slot_active = new_outflow[:, k] & (d_ids >= 0) & can_spill
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
                waste_this_tick += WASTE_WEIGHT_DEST_TERMINATED * float(
                    per_edge[terminated_src].sum()
                )

    s.owner = new_owner
    s.strength = new_strength
    s.outflow = new_outflow
    s.edge_pressure = new_edge_pressure.astype(np.float32)
    s.tick = state.tick + 1
    s.waste_total = state.waste_total + waste_this_tick

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
