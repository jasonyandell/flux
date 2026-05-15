"""MLX-batched v2 step.

Same algorithm as the pure reducer in `step.py`, but operating on a batch of
G games in parallel for trainer throughput. The persistent `edge_pressure`
tensor is part of batched state, exactly like the PRD calls out.

Convention:
  owner          (G, N)        int32   NEUTRAL=-1, DEAD=-2, else seat 0..P-1
  strength       (G, N)        float32
  outflow        (G, N, K)     bool
  edge_pressure  (G, N, K)     float32
  neighbors      (N, K)        int32   slot k → cell id (shared across batch)
  alive          (G,)          bool    games still being stepped
"""
from __future__ import annotations

import mlx.core as mx

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
    REGEN_BASE_PER_TICK,
    REGEN_SLOPE,
    TRANSIT_CREDIT_STRICT,
    WASTE_WEIGHT_CAP_BOUND,
    WASTE_WEIGHT_DEST_TERMINATED,
    WASTE_WEIGHT_NO_SPILL,
)


# ----- helpers --------------------------------------------------------------


def _gather_neighbor(arr: mx.array, neighbors: mx.array) -> mx.array:
    """Gather along axis 1 according to neighbors.
    arr: (G, N) or (G, N, K). neighbors: (N, K) int. Returns:
      - if arr (G, N): (G, N, K) of arr[:, neighbors[c, k]]
      - if arr (G, N, K): (G, N, K, K) which we don't use here.
    -1 entries return arr at index 0 (caller masks them).
    """
    G, N = arr.shape[:2]
    safe = mx.maximum(neighbors, 0)             # (N, K)
    flat = safe.reshape(-1)                     # (N*K,)
    out = mx.take(arr, flat, axis=1)            # (G, N*K, ...)
    new_shape = (G, N, K) + arr.shape[2:]
    return out.reshape(new_shape)


# ----- action application ---------------------------------------------------


def apply_actions_batched(
    owner: mx.array,
    outflow: mx.array,
    actions: mx.array,
    neighbors: mx.array,
) -> mx.array:
    """Apply per-cell actions to outflow.

    actions: (G, N) int32 — action codes per cell.
    Returns new_outflow (G, N, K) bool, with invariants resolved:
      - SET on slot with off-grid neighbor: no-op
      - SET/CLEAR on neutral or dead cell: no-op
      - Friendly bidirectional: lower cell-index loses (higher wins).
    """
    G, N = owner.shape
    is_owned = (owner >= 0)                                     # (G, N)
    nb_valid_per_cell = (neighbors >= 0)                        # (N, K)

    # Neighbor owner per (g, c, k); dead neighbors block SET (forbidden
    # connection target).
    nb_safe_for_owner = mx.maximum(neighbors, 0)                # (N, K)
    nb_owner_pre = mx.take(owner, nb_safe_for_owner.reshape(-1), axis=1).reshape(G, N, K)
    nb_is_dead = (nb_owner_pre == DEAD)                          # (G, N, K)

    # Decode action.
    set_mask = (actions >= ACTION_SET_BASE) & (actions < ACTION_CLEAR_BASE)
    clear_mask = (actions >= ACTION_CLEAR_BASE) & (actions < ACTION_NOOP)
    # Per-slot SET / CLEAR masks: (G, N, K).
    slot_idx = mx.arange(K).reshape(1, 1, K)
    set_per_slot = (
        set_mask.reshape(G, N, 1)
        & (actions.reshape(G, N, 1) == slot_idx)
        & is_owned.reshape(G, N, 1)
        & nb_valid_per_cell.reshape(1, N, K)
        & mx.logical_not(nb_is_dead)
    )
    clear_per_slot = (
        clear_mask.reshape(G, N, 1)
        & ((actions - ACTION_CLEAR_BASE).reshape(G, N, 1) == slot_idx)
        & is_owned.reshape(G, N, 1)
        & nb_valid_per_cell.reshape(1, N, K)
    )
    new_outflow = mx.where(set_per_slot, True, outflow)
    new_outflow = mx.where(clear_per_slot, False, new_outflow)

    # Resolve friendly bidirectional. For each (g, c, k): if new_outflow[g, c, k]
    # is True and neighbors[c, k] = d ≥ 0, look at new_outflow[g, d, opp_k] and
    # owner[g, d]. If both set and friendly, clear the lower cell-index side.
    opp = OPPOSITE_SLOT                                              # (K,)
    opp_mx = mx.array(opp.tolist(), dtype=mx.int32)
    # neighbor_d[c, k] = neighbors[c, k]
    nb_safe = mx.maximum(neighbors, 0)                              # (N, K)
    # Gather: for each (G, c, k), get owner[g, d] and outflow_back[g, d, opp_k].
    # owner_d[g, c, k] = owner[g, neighbors[c, k]]
    owner_d = mx.take(owner, nb_safe.reshape(-1), axis=1).reshape(G, N, K)
    nb_valid = nb_valid_per_cell.reshape(1, N, K)
    is_friendly_d = (owner_d == owner.reshape(G, N, 1)) & nb_valid & is_owned.reshape(G, N, 1)

    # Back-edge: outflow[g, d, opp_k]. We need to gather along axis 1 by d,
    # then index slot dimension by opp_k. Build a flat index per (c, k).
    # Resulting shape (G, N, K) of bools.
    # Approach: build a per-(c, k) flat index into outflow_flat (G, N*K).
    flat_d_slot = (nb_safe * K + opp_mx.reshape(1, K)).reshape(-1)  # (N*K,)
    of_flat = new_outflow.reshape(G, N * K)
    back_outflow = mx.take(of_flat, flat_d_slot, axis=1).reshape(G, N, K)

    conflict = new_outflow & back_outflow & is_friendly_d
    # Lower cell-index loses: clear new_outflow[g, c, k] where c < d.
    cell_idx = mx.arange(N).reshape(1, N, 1).astype(mx.int32)
    nb_safe_b = nb_safe.reshape(1, N, K)
    c_lower = cell_idx < nb_safe_b
    clear_loser = conflict & c_lower
    new_outflow = mx.where(clear_loser, False, new_outflow)
    return new_outflow


# ----- per-tick physics -----------------------------------------------------


def _regen(strength: mx.array) -> mx.array:
    return REGEN_BASE_PER_TICK * (1.0 + REGEN_SLOPE * (strength - 1.0))


def tick_batched(
    owner: mx.array,
    strength: mx.array,
    outflow: mx.array,
    edge_pressure: mx.array,
    alive: mx.array,
    num_players: int,
    neighbors: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Apply one game tick to the batch.

    Returns (new_owner, new_strength, new_outflow, new_edge_pressure,
             waste_per_cell, transit_credit_per_cell). For dead games
    (`alive`==False) we keep all tensors unchanged.
    """
    G, N = owner.shape
    P = num_players
    is_owned = (owner >= 0)                                          # (G, N)
    is_dead = (owner == DEAD)
    is_neutral = (owner == NEUTRAL)
    alive_f = alive.astype(mx.float32).reshape(G, 1)

    # --- gather inbound pressure from each neighbor's back-edge ---
    # For slot k at cell c, source is neighbors[c, k] = d and slot d→c = OPP[k].
    nb_safe = mx.maximum(neighbors, 0)                                # (N, K)
    nb_valid = (neighbors >= 0).reshape(1, N, K)
    opp_mx = mx.array(OPPOSITE_SLOT.tolist(), dtype=mx.int32)         # (K,)
    flat_d_slot = (nb_safe * K + opp_mx.reshape(1, K)).reshape(-1)    # (N*K,)
    ep_flat = edge_pressure.reshape(G, N * K)
    pressure_in_per_slot = mx.take(ep_flat, flat_d_slot, axis=1).reshape(G, N, K)
    # Only count edges if source d has outflow[d, opp_k] set (otherwise the
    # pressure stored is 0 by construction, but enforce defensively).
    of_flat = outflow.reshape(G, N * K)
    src_outflow = mx.take(of_flat, flat_d_slot, axis=1).reshape(G, N, K)
    pressure_in_per_slot = pressure_in_per_slot * src_outflow.astype(mx.float32)
    # Apply validity.
    pressure_in_per_slot = pressure_in_per_slot * nb_valid.astype(mx.float32)

    # Partition by friendly vs enemy.
    owner_d = mx.take(owner, nb_safe.reshape(-1), axis=1).reshape(G, N, K)
    is_friendly = (owner_d == owner.reshape(G, N, 1)) & nb_valid & is_owned.reshape(G, N, 1)
    is_enemy   = (owner_d != owner.reshape(G, N, 1)) & (owner_d >= 0) & nb_valid

    pressure_in_friendly_nb = (pressure_in_per_slot * is_friendly.astype(mx.float32)).sum(axis=-1)
    pressure_in_enemy_nb    = (pressure_in_per_slot * is_enemy.astype(mx.float32)).sum(axis=-1)

    # Regen contributes to friendly pressure on owned cells.
    regen_amount = _regen(strength) * is_owned.astype(mx.float32)
    pressure_in_friendly = pressure_in_friendly_nb + regen_amount
    pressure_in_enemy = pressure_in_enemy_nb

    # --- Fill / overflow / capture detect ---
    headroom = mx.maximum(MAX_STRENGTH - strength, 0.0)
    grew = mx.minimum(pressure_in_friendly, headroom)
    overflow = pressure_in_friendly - grew
    overflow = overflow * is_owned.astype(mx.float32)
    pre_strength = strength + grew - pressure_in_enemy

    # Captures.
    # Attribute enemy pressure by source player (G, N, P) to pick winning attacker.
    src_player = mx.clip(owner_d, 0, P - 1)                          # safe
    src_one_hot = mx.zeros((G, N, K, P), dtype=mx.float32)
    p_idx = mx.arange(P).reshape(1, 1, 1, P)
    src_one_hot = (src_player.reshape(G, N, K, 1) == p_idx).astype(mx.float32)
    src_one_hot = src_one_hot * is_enemy.reshape(G, N, K, 1).astype(mx.float32)
    enemy_pressure_by_player = (
        pressure_in_per_slot.reshape(G, N, K, 1) * src_one_hot
    ).sum(axis=2)                                                     # (G, N, P)

    will_capture_owned = (pre_strength < 0) & is_owned & (pressure_in_enemy > 0)
    neutral_pre = strength - pressure_in_enemy
    will_capture_neutral = (neutral_pre < 0) & is_neutral & (pressure_in_enemy > 0)
    capture_mask = will_capture_owned | will_capture_neutral

    best_attacker = enemy_pressure_by_player.argmax(axis=-1).astype(owner.dtype)  # (G, N)

    new_owner = mx.where(capture_mask, best_attacker, owner)
    new_owner = mx.where(is_dead, mx.array(DEAD, dtype=owner.dtype), new_owner)

    new_strength_owned = mx.clip(pre_strength, 0.0, MAX_STRENGTH)
    new_strength_neutral = mx.clip(neutral_pre, 0.0, MAX_STRENGTH)
    new_strength = mx.where(
        is_owned, new_strength_owned, mx.where(is_neutral, new_strength_neutral, strength)
    )
    # New big-bag-of-pressure rule: captured cell takes |pre_strength| as
    # starting strength — the surplus pressure that broke the defender.
    surplus = mx.minimum(mx.maximum(-pre_strength, 0.0), MAX_STRENGTH)
    new_strength = mx.where(capture_mask, surplus, new_strength)
    new_strength = mx.where(is_dead, mx.array(0.0, dtype=mx.float32), new_strength)

    # --- Spill ---
    # Capturing cells lose their outflow.
    capture_b = capture_mask.reshape(G, N, 1)
    new_outflow = mx.where(capture_b, False, outflow)

    of_f = new_outflow.astype(mx.float32)
    num_active = of_f.sum(axis=-1)                                    # (G, N)
    can_spill = (num_active > 0) & (overflow > 0) & is_owned

    per_edge_unclipped = overflow / mx.maximum(num_active, 1.0)       # safe div
    per_edge = mx.minimum(per_edge_unclipped, MAX_EDGE)
    # New edge_pressure: zero where capture cleared outflow; else per_edge on active slots.
    new_edge_pressure = of_f * per_edge.reshape(G, N, 1)
    # Apply live-mask: dead games keep their old edge_pressure.
    alive_b = alive.reshape(G, 1, 1).astype(mx.float32)
    new_edge_pressure = mx.where(alive.reshape(G, 1, 1), new_edge_pressure, edge_pressure)

    # Waste: cap-bound excess + overflow with no outflows. Weighted to make
    # "no outflow + overflowing" (lazy waste) qualitatively worse than
    # cap-bound (busy waste). See WASTE_WEIGHT_NO_SPILL / WASTE_WEIGHT_CAP_BOUND.
    waste_cap_bound = (per_edge_unclipped - per_edge) * num_active
    waste_cap_bound = WASTE_WEIGHT_CAP_BOUND * mx.where(
        can_spill, waste_cap_bound, mx.array(0.0, dtype=mx.float32),
    )
    no_spill = is_owned & (overflow > 0) & (num_active == 0)
    waste_no_spill = WASTE_WEIGHT_NO_SPILL * mx.where(
        no_spill, overflow, mx.array(0.0, dtype=mx.float32),
    )
    waste_per_cell = waste_cap_bound + waste_no_spill

    # Destination-terminated waste: active outflow points at a friendly cell
    # at MAX strength with zero active outflows (pure sink). Pass-through MAX
    # cells with outflows are combo relays — only penalize dead ends.
    if WASTE_WEIGHT_DEST_TERMINATED > 0.0:
        strength_d = mx.take(strength, nb_safe.reshape(-1), axis=1).reshape(G, N, K)
        num_active_d = mx.take(num_active, nb_safe.reshape(-1), axis=1).reshape(G, N, K)
        is_dead_end = (
            is_friendly
            & (strength_d >= MAX_STRENGTH)
            & (num_active_d == 0.0)
        )
        dest_terminated = (
            of_f
            * is_dead_end.astype(mx.float32)
            * can_spill.reshape(G, N, 1).astype(mx.float32)
            * per_edge.reshape(G, N, 1)
        ).sum(axis=-1)
        waste_per_cell = waste_per_cell + WASTE_WEIGHT_DEST_TERMINATED * dest_terminated

    # Transit credit: positive local signal for pressure sent into a friendly
    # relay. Strict mode pays only MAX-strength relays, mirroring the
    # dest-terminated sink rule without rewarding ordinary fill traffic.
    strength_d = mx.take(strength, nb_safe.reshape(-1), axis=1).reshape(G, N, K)
    num_active_d = mx.take(num_active, nb_safe.reshape(-1), axis=1).reshape(G, N, K)
    is_relay = is_friendly & (num_active_d > 0.0)
    if TRANSIT_CREDIT_STRICT:
        is_relay = is_relay & (strength_d >= MAX_STRENGTH)
    transit_credit_per_cell = (
        of_f
        * is_relay.astype(mx.float32)
        * can_spill.reshape(G, N, 1).astype(mx.float32)
        * per_edge.reshape(G, N, 1)
    ).sum(axis=-1)

    # Apply alive_mask: frozen games keep old state.
    alive_mask_b = alive.reshape(G, 1)
    new_owner_final = mx.where(alive_mask_b, new_owner, owner)
    new_strength_final = mx.where(alive_mask_b, new_strength, strength)
    new_outflow_final = mx.where(alive.reshape(G, 1, 1), new_outflow, outflow)
    waste_per_cell = waste_per_cell * alive.reshape(G, 1).astype(mx.float32)
    transit_credit_per_cell = (
        transit_credit_per_cell * alive.reshape(G, 1).astype(mx.float32)
    )

    return (new_owner_final, new_strength_final, new_outflow_final,
            new_edge_pressure, waste_per_cell, transit_credit_per_cell)
