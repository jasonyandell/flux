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
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Returns (new_owner, new_strength, new_flow_valid, new_passthrough)."""
    G, N = owner.shape
    P = num_players
    live_f32 = live_mask.astype(mx.float32).reshape(G, 1)
    is_owned = (owner != -1).astype(mx.float32)
    owner_safe = mx.maximum(owner, 0)

    flow_dst_safe = mx.maximum(flow_dst, 0)
    flow_player_safe = mx.maximum(flow_player, 0)

    is_sending = flow_valid & (owner != -1) & (owner == flow_player)
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

    # Sending cells: deliver output_capacity * dt to dst (sign by friendly/enemy).
    dst_owner = mx.take_along_axis(owner, flow_dst_safe, axis=1)
    is_friendly_dst = ((dst_owner == flow_player) & is_sending).astype(mx.float32)
    is_enemy_dst = ((dst_owner != flow_player) & is_sending).astype(mx.float32)
    flux = output_capacity_rate * dt * live_f32             # (G, N)
    in_delta = flux * (is_friendly_dst - is_enemy_dst)

    dst_idx = (G_idx * (N * P) + flow_dst_safe * P + flow_player_safe).reshape(-1)
    forces_flat = forces_flat.at[dst_idx].add(in_delta.reshape(-1))
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
    computed_strength = mx.minimum(computed_strength, float(MAX_STRENGTH))

    # Frozen games: keep prior state. Captured cells reset their passthrough.
    live_b = live_mask.reshape(G, 1)
    new_owner = mx.where(live_b, computed_owner, owner)
    new_strength_final = mx.where(live_b, computed_strength, strength)
    new_flow_valid = is_sending & live_b
    # Cells whose owner changed (captured) should drop their passthrough.
    owner_unchanged = (new_owner == owner).astype(mx.float32)
    new_passthrough = next_passthrough * owner_unchanged * live_f32

    return new_owner, new_strength_final, new_flow_valid, new_passthrough
