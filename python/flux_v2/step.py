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
    K,
    MAX_EDGE,
    MAX_STRENGTH,
    NEUTRAL,
    OPPOSITE_SLOT,
    State,
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
    """Apply one action per cell (per the policy's output).

    `actions` is (N,) int. Action 0..K-1 = set slot k; K..2K-1 = clear slot
    k-K; 2K = no-op. Actions on cells the policy doesn't own are still
    accepted at the reducer level — the trainer should only feed actions for
    owned cells, but unowned cells are no-op'd here so the reducer stays
    pure even if a stale action sneaks through.

    Mutation invariants resolved here:
      1. No friendly bidirectional flow.
      2. Higher cell-index wins simultaneous mutual mutation.
      3. (Capture clears) — handled in `tick`, not here.
      4. Stale targets stay on — handled implicitly: only actions on the
         current owner's cells take effect.
    """
    s = copy_state(state)
    N = s.N
    owner = s.owner
    nb = s.neighbors

    if actions.shape != (N,):
        raise ValueError(f"actions shape mismatch: {actions.shape} vs ({N},)")

    # First pass: per-cell, apply the mutation (no cross-cell coupling yet).
    # We track which cells turned ON a slot toward a *friendly* neighbor; those
    # need invariant resolution before commit.
    new_outflow = s.outflow.copy()
    # (cell, slot) pairs that flipped on this tick targeting a friendly neighbor.
    pending_on: list[tuple[int, int]] = []

    for c in range(N):
        a = int(actions[c])
        if a == ACTION_NOOP:
            continue
        if owner[c] < 0:                # neutral or dead — can't act
            continue
        if ACTION_SET_BASE <= a < ACTION_CLEAR_BASE:
            k = a - ACTION_SET_BASE
            if nb[c, k] < 0:           # off-grid; no-op
                continue
            d = int(nb[c, k])
            if owner[d] == DEAD:       # connections to dead cells are forbidden
                continue
            if new_outflow[c, k]:
                continue               # idempotent
            new_outflow[c, k] = True
            if owner[d] == owner[c]:
                pending_on.append((c, k))
        elif ACTION_CLEAR_BASE <= a < ACTION_NOOP:
            k = a - ACTION_CLEAR_BASE
            if nb[c, k] < 0:
                continue
            new_outflow[c, k] = False

    # Invariant resolution:
    #  - For each freshly-set c→d with d friendly:
    #     if d→c is set, clear the lower-cell-index side.
    for (c, k) in pending_on:
        d = int(nb[c, k])
        if owner[d] != owner[c]:
            continue                   # ownership might have changed (not really, but safe)
        opp = int(OPPOSITE_SLOT[k])
        if not new_outflow[d, opp]:
            continue                   # no back-edge — fine
        # Back-edge is set. Higher-cell-index wins; clear the loser.
        if c > d:
            new_outflow[d, opp] = False
        else:
            new_outflow[c, k] = False

    s.outflow = new_outflow
    return s


# ---------------------------------------------------------------------------
# Per-tick physics
# ---------------------------------------------------------------------------

def tick(state: State) -> State:
    """Apply the per-tick algorithm. Returns a fresh state. Edge pressure
    recomputed from the cell's overflow this tick.
    """
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

    # ---- Gather pressure_in_friendly and pressure_in_enemy at each cell ----
    pressure_in_friendly = np.zeros(N, dtype=np.float32)
    pressure_in_enemy = np.zeros(N, dtype=np.float32)
    enemy_pressure_by_player = np.zeros((N, max(s.num_players, 1)), dtype=np.float32)

    # Regen contributes to friendly pressure only on owned cells.
    pressure_in_friendly[is_alive] = regen(strength[is_alive]).astype(np.float32)

    # For each neighbor slot k of each cell c, look at edge_pressure[d, slot_d→c]
    # where d = neighbors[c, k].  d → c slot is OPPOSITE_SLOT[k].
    for k in range(K):
        d_ids = nb[:, k]                                   # (N,)
        valid = d_ids >= 0
        opp = int(OPPOSITE_SLOT[k])
        # Pressure flowing from d into c along slot opp.
        # Only counts if d had outflow[d, opp] = True (otherwise edge_pressure[d, opp] is 0).
        pressure = np.zeros(N, dtype=np.float32)
        valid_d = d_ids[valid]
        pressure[valid] = edge_pressure_prev[valid_d, opp]
        # Partition by friendly vs enemy (relative to receiver c).
        d_owner = np.full(N, NEUTRAL, dtype=np.int32)
        d_owner[valid] = owner[valid_d]
        is_friendly_d = valid & (d_owner == owner) & is_alive
        is_enemy_d = valid & (d_owner != owner) & (d_owner >= 0) & is_alive
        pressure_in_friendly += pressure * is_friendly_d.astype(np.float32)
        pressure_in_enemy   += pressure * is_enemy_d.astype(np.float32)
        # Attribute enemy pressure by the source player so we can resolve
        # captures (which attacker takes the cell).
        if is_enemy_d.any():
            idxs = np.where(is_enemy_d)[0]
            attackers = d_owner[idxs]
            enemy_pressure_by_player[idxs, attackers] += pressure[idxs]

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
        new_strength[cap_cells] = CAPTURE_STRENGTH

    # Dead cells stay dead.
    new_owner[is_dead] = DEAD
    new_strength[is_dead] = 0.0

    # ---- Spill (rewrite edge_pressure for next tick) ----
    new_edge_pressure = np.zeros_like(edge_pressure_prev)
    waste_this_tick = 0.0

    # Block 1: capturing cells lose their outflows.
    new_outflow = outflow.copy()
    if capture_mask.any():
        new_outflow[capture_mask] = False

    num_active = new_outflow.sum(axis=1).astype(np.int32)              # (N,)
    can_spill = (num_active > 0) & (overflow > 0) & is_alive

    if can_spill.any():
        per_edge_unclipped = np.zeros(N, dtype=np.float32)
        per_edge_unclipped[can_spill] = overflow[can_spill] / num_active[can_spill]
        per_edge = np.minimum(per_edge_unclipped, MAX_EDGE)
        # Write pressure on active slots.
        # Broadcast per-cell scalar to the active-slot mask.
        new_edge_pressure = new_outflow * per_edge[:, None]
        # Cap-bound waste: amount per slot above MAX_EDGE (busy waste).
        excess_per_slot = (per_edge_unclipped - per_edge)             # (N,)
        waste_this_tick += WASTE_WEIGHT_CAP_BOUND * float(
            (excess_per_slot[can_spill] * num_active[can_spill]).sum()
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
