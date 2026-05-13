"""Batched MLX implementation of the regen-flow step.

Semantics (see decisions/regen-flow-rules.md):
  • Per-cell regen rate scales linearly with strength.
  • Sending cells forfeit their regen — it's redirected as outgoing flux,
    split evenly across the cell's outflows.
  • Damage is symmetric: friendly destinations gain `flux`, enemies lose it.
  • Capture flips ownership and sets strength = CAPTURE_STRENGTH (=1.0).
  • Strength capped at MAX_STRENGTH.

Passthrough (1-tick-lagged):
  • A sending cell's output_capacity = regen(s) + passthrough_carry, where
    passthrough_carry is the friendly support that cell received last tick.
  • At end of tick, next_passthrough_carry = support_in (this tick), gated
    by is_sending (cells that idle bank support directly; cells that send
    relay the support to next-tick's output, with one-tick lag).
  • Capped per outflow at MAX_OUTPUT_PER_SEC so a long chain can't fire
    an arbitrarily-large insta-kill bolt.

The flow representation here is one outflow slot per cell (G, N), matching
`build_flows_from_actions`. In that representation each cell has K ∈ {0, 1}
outflows.
"""
from __future__ import annotations

import mlx.core as mx

from .state import MAX_STRENGTH

REGEN_BASE_PER_SEC: float = 0.5
REGEN_SLOPE: float = 2.0
CAPTURE_STRENGTH: float = 1.0
MAX_OUTPUT_PER_SEC: float = float(MAX_STRENGTH)  # insta-kill cap per outflow


def cell_regen_rate(strength: mx.array) -> mx.array:
    """Linear regen scaling: regen(s) = base * (1 + slope * (s - 1))."""
    return REGEN_BASE_PER_SEC * (1.0 + REGEN_SLOPE * (strength - 1.0))


def init_passthrough(G: int, N: int) -> mx.array:
    return mx.zeros((G, N), dtype=mx.float32)


STRIDE: int = 18  # neighbor-table stride; matches NEIGHBOR_STRIDE in mlx_batch
MAXED_FANOUT_THRESHOLD: float = float(MAX_STRENGTH)  # cells at >= this auto-fanout


def step_batched_regen(
    owner: mx.array,           # (G, N) int32, -1 = neutral
    strength: mx.array,        # (G, N) float32
    flow_src: mx.array,        # (G, N) int32 — slot i has src = i (or -1)
    flow_dst: mx.array,        # (G, N) int32
    flow_player: mx.array,     # (G, N) int32
    flow_valid: mx.array,      # (G, N) bool
    live_mask: mx.array,       # (G,) bool
    num_players: int,
    dt: float,
    passthrough: mx.array,     # (G, N) float32 — last tick's support_in for sending cells
    neighbors: mx.array | None = None,  # (N * STRIDE,) int32 — required for maxed-fanout
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Returns (new_owner, new_strength, new_flow_valid, new_passthrough).

    Maxed cells (strength >= MAXED_FANOUT_THRESHOLD) auto-fanout: instead of the
    single action-chosen outflow, they distribute output_capacity across ALL
    valid neighbors evenly. The action-driven single outflow is suppressed for
    those cells.
    """
    G, N = owner.shape
    P = num_players
    live_f32 = live_mask.astype(mx.float32).reshape(G, 1)
    is_owned = (owner != -1).astype(mx.float32)
    owner_safe = mx.maximum(owner, 0)

    flow_dst_safe = mx.maximum(flow_dst, 0)
    flow_player_safe = mx.maximum(flow_player, 0)

    # Maxed-cell fanout: any owned cell at MAXED_FANOUT_THRESHOLD becomes a
    # sender regardless of its action. The single-action flow is suppressed.
    is_maxed_owned = (strength >= MAXED_FANOUT_THRESHOLD) & (owner != -1)
    is_action_sending = flow_valid & (owner != -1) & (owner == flow_player)
    # Single-action sending applies only to non-maxed cells.
    is_single_sending = is_action_sending & mx.logical_not(is_maxed_owned)
    is_sending = is_single_sending | is_maxed_owned
    is_sending_f32 = is_sending.astype(mx.float32)
    is_idle_owned_f32 = is_owned * (1.0 - is_sending_f32)

    cell_regen = cell_regen_rate(strength)                 # (G, N) per-second
    # Output capacity for sending cells = regen + carried passthrough; capped.
    output_capacity_rate = mx.minimum(
        cell_regen + passthrough, MAX_OUTPUT_PER_SEC,
    )                                                       # (G, N)

    forces_flat = mx.zeros(G * N * P, dtype=mx.float32)
    G_idx = mx.arange(G).reshape(G, 1)
    N_idx = mx.arange(N).reshape(1, N)
    self_idx = (G_idx * (N * P) + N_idx * P + owner_safe).reshape(-1)

    # Idle cells: regen goes to their own slot.
    self_delta = cell_regen * dt * is_idle_owned_f32 * live_f32
    forces_flat = forces_flat.at[self_idx].add(self_delta.reshape(-1))

    # Single-action sending cells: deliver output_capacity * dt to dst.
    is_single_sending_f32 = is_single_sending.astype(mx.float32)
    dst_owner = mx.take_along_axis(owner, flow_dst_safe, axis=1)
    is_friendly_dst = ((dst_owner == flow_player) & is_single_sending).astype(mx.float32)
    is_enemy_dst = ((dst_owner != flow_player) & is_single_sending).astype(mx.float32)
    flux_single = output_capacity_rate * dt * live_f32      # (G, N)
    in_delta = flux_single * (is_friendly_dst - is_enemy_dst)

    dst_idx = (G_idx * (N * P) + flow_dst_safe * P + flow_player_safe).reshape(-1)
    forces_flat = forces_flat.at[dst_idx].add(in_delta.reshape(-1))

    # Maxed-cell fanout: distribute output_capacity across all valid neighbors
    # that can actually absorb it (enemies, or friends with headroom). Skipping
    # maxed friends prevents wasted bidirectional flow between two maxed cells.
    if neighbors is not None:
        is_maxed_f32 = is_maxed_owned.astype(mx.float32)            # (G, N)
        nb_ids = neighbors.reshape(N, STRIDE).astype(mx.int32)      # (N, STRIDE)
        nb_valid_per_cell = (nb_ids >= 0).astype(mx.float32)        # (N, STRIDE)
        nb_valid_b = nb_valid_per_cell.reshape(1, N, STRIDE)

        safe_nb = mx.maximum(nb_ids, 0).reshape(-1)                 # (N*STRIDE,)
        nb_owner = mx.take(owner, safe_nb, axis=1).reshape(G, N, STRIDE)
        nb_strength = mx.take(strength, safe_nb, axis=1).reshape(G, N, STRIDE)
        src_owner_b = owner_safe.reshape(G, N, 1)
        nb_is_friend = (nb_owner == src_owner_b)
        nb_is_maxed = nb_strength >= MAXED_FANOUT_THRESHOLD
        # A neighbor is a "useful target" if it's valid AND not a maxed friend.
        useful_mask_f32 = (nb_valid_b.astype(mx.bool_)
                          & mx.logical_not(nb_is_friend & nb_is_maxed)).astype(mx.float32)
        num_useful = useful_mask_f32.sum(axis=-1)                   # (G, N)
        # If no useful targets, fanout share is 0 (no outflow this tick).
        share = mx.where(num_useful > 0, 1.0 / mx.maximum(num_useful, 1.0), 0.0)
        flux_max = output_capacity_rate * is_maxed_f32 * dt * live_f32   # (G, N)
        per_nb_flux = (flux_max * share).reshape(G, N, 1) * useful_mask_f32  # (G, N, STRIDE)

        is_friendly = nb_is_friend.astype(mx.float32)
        is_enemy = (mx.logical_not(nb_is_friend)).astype(mx.float32) * nb_valid_b
        sign = is_friendly - is_enemy                                # (G, N, STRIDE)
        contrib = per_nb_flux * sign                                # (G, N, STRIDE)
        nb_flat_idx = (G_idx.reshape(G, 1, 1) * (N * P)
                       + mx.maximum(nb_ids, 0).reshape(1, N, STRIDE) * P
                       + owner_safe.reshape(G, N, 1)).reshape(-1)
        forces_flat = forces_flat.at[nb_flat_idx].add(contrib.reshape(-1))

    forces = forces_flat.reshape(G, N, P)

    # ---- support_in (this tick's friendly inflow per cell, in rate units) ----
    # forces[c, owner_c] = regen(if idle) * dt + friendly_inflows * dt
    # So we subtract the regen contribution to get pure friendly support.
    own_force = mx.take_along_axis(
        forces, owner_safe.reshape(G, N, 1), axis=2,
    ).reshape(G, N)
    own_force = own_force * is_owned                        # zero for neutrals
    self_regen_contribution = cell_regen * dt * is_idle_owned_f32
    support_in_amount = mx.maximum(own_force - self_regen_contribution, 0.0)
    # Convert from per-tick to per-second so it composes cleanly with cell_regen.
    support_in_rate = support_in_amount / max(dt, 1e-9)     # (G, N)
    # Only sending cells carry passthrough forward (idle cells bank it).
    next_passthrough = support_in_rate * is_sending_f32

    # ---- Strength update ----
    net_delta = forces.sum(axis=2)                          # (G, N)
    new_strength = strength + net_delta

    # ---- Capture detection ----
    one_hot = mx.zeros((G, N, P), dtype=mx.float32)
    one_hot_flat = one_hot.reshape(-1).at[self_idx].add(is_owned.reshape(-1))
    one_hot = one_hot_flat.reshape(G, N, P)
    damages_no_owner = (-forces) * (1.0 - one_hot)
    best_damage = damages_no_owner.max(axis=2)
    best_player = damages_no_owner.argmax(axis=2).astype(owner.dtype)

    negative = new_strength < 0
    takeover = negative & (best_damage > 0)
    capture_strength_arr = mx.array(CAPTURE_STRENGTH, dtype=mx.float32)
    computed_owner = mx.where(takeover, best_player, owner)
    computed_strength = mx.where(
        takeover,
        capture_strength_arr,
        mx.maximum(new_strength, 0.0),
    )
    # Friendly inflow that would have pushed strength over the cap is wasted.
    # We attribute it to the cell's owner so the reward loop can penalize it.
    pre_clip = mx.where(takeover, capture_strength_arr, mx.maximum(new_strength, 0.0))
    wasted_per_cell = mx.maximum(pre_clip - float(MAX_STRENGTH), 0.0) * is_owned
    computed_strength = mx.minimum(computed_strength, float(MAX_STRENGTH))

    # Frozen games: keep prior state. Captured cells reset their passthrough.
    live_b = live_mask.reshape(G, 1)
    new_owner = mx.where(live_b, computed_owner, owner)
    new_strength_final = mx.where(live_b, computed_strength, strength)
    new_flow_valid = is_sending & live_b
    # Cells whose owner changed (captured) should drop their passthrough.
    owner_unchanged = (new_owner == owner).astype(mx.float32)
    new_passthrough = next_passthrough * owner_unchanged * live_f32
    wasted_per_cell = wasted_per_cell * live_f32

    return new_owner, new_strength_final, new_flow_valid, new_passthrough, wasted_per_cell
