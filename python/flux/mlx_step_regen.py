"""Batched MLX implementation of the regen-flow step.

Semantics (see decisions/regen-flow-rules.md):
  • Per-cell regen rate scales linearly with strength.
  • Sending cells forfeit their regen — it's redirected as outgoing flux,
    split evenly across the cell's outflows.
  • Damage is symmetric: friendly destinations gain `flux`, enemies lose it.
  • Capture flips ownership and sets strength = CAPTURE_STRENGTH (=1.0).
  • Strength capped at MAX_STRENGTH (overage propagation TBD).

The flow representation here is one outflow slot per cell (G, N), matching
`build_flows_from_actions`. In that representation each cell has K ∈ {0, 1}
outflows, so the split-across-K simplifies to "full regen if sending, no
regen if not." General multi-outflow K > 1 is supported by the TS path
(step_regen.ts) but not exercised by training.
"""
from __future__ import annotations

import mlx.core as mx

from .state import MAX_STRENGTH

REGEN_BASE_PER_SEC: float = 0.5
REGEN_SLOPE: float = 2.0
CAPTURE_STRENGTH: float = 1.0


def cell_regen_rate(strength: mx.array) -> mx.array:
    """Linear regen scaling: regen(s) = base * (1 + slope * (s - 1))."""
    return REGEN_BASE_PER_SEC * (1.0 + REGEN_SLOPE * (strength - 1.0))


def step_batched_regen(
    owner: mx.array,        # (G, N) int32, -1 = neutral
    strength: mx.array,     # (G, N) float32
    flow_src: mx.array,     # (G, N) int32 — slot i has src = i (or -1 if invalid)
    flow_dst: mx.array,     # (G, N) int32 — neighbor of cell i, or -1
    flow_player: mx.array,  # (G, N) int32 — flow.player; equal to owner of slot i when valid
    flow_valid: mx.array,   # (G, N) bool
    live_mask: mx.array,    # (G,) bool — frozen games are skipped
    num_players: int,
    dt: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Returns (new_owner, new_strength, new_flow_valid)."""
    G, N = owner.shape
    P = num_players
    live_f32 = live_mask.astype(mx.float32).reshape(G, 1)
    is_owned = (owner != -1).astype(mx.float32)            # (G, N)
    owner_safe = mx.maximum(owner, 0)

    flow_dst_safe = mx.maximum(flow_dst, 0)
    flow_player_safe = mx.maximum(flow_player, 0)

    # Cell i is sending iff it has a valid flow slot AND owns itself AND the
    # flow's player matches. (flow_player is derived from owner upstream, but
    # we re-check defensively.)
    is_sending = flow_valid & (owner != -1) & (owner == flow_player)
    is_sending_f32 = is_sending.astype(mx.float32)
    is_idle_owned_f32 = is_owned * (1.0 - is_sending_f32)

    # Per-cell regen capacity (in strength-units per second, before dt).
    cell_regen = cell_regen_rate(strength)                 # (G, N)

    # Force accumulator (G, N, P), flattened for scatter-add.
    forces_flat = mx.zeros(G * N * P, dtype=mx.float32)
    G_idx = mx.arange(G).reshape(G, 1)
    N_idx = mx.arange(N).reshape(1, N)

    # Idle cells: regen goes to their own slot.
    self_idx = (G_idx * (N * P) + N_idx * P + owner_safe).reshape(-1)
    self_delta = cell_regen * dt * is_idle_owned_f32 * live_f32
    forces_flat = forces_flat.at[self_idx].add(self_delta.reshape(-1))

    # Sending cells: full regen routed to their one outflow's destination,
    # credited to the flow.player slot at the dst cell.
    dst_owner = mx.take_along_axis(owner, flow_dst_safe, axis=1)
    is_friendly_dst = ((dst_owner == flow_player) & is_sending).astype(mx.float32)
    is_enemy_dst = ((dst_owner != flow_player) & is_sending).astype(mx.float32)
    flux = cell_regen * dt * live_f32                      # (G, N), per-cell capacity
    in_delta = flux * (is_friendly_dst - is_enemy_dst)     # + for support, − for attack

    dst_idx = (G_idx * (N * P) + flow_dst_safe * P + flow_player_safe).reshape(-1)
    forces_flat = forces_flat.at[dst_idx].add(in_delta.reshape(-1))
    forces = forces_flat.reshape(G, N, P)

    # Net strength delta = sum across players (owner-side is positive, enemies negative).
    net_delta = forces.sum(axis=2)                          # (G, N)
    new_strength = strength + net_delta

    # Capture detection: damages[g, n, p] = max(0, -forces[g, n, p]).
    # Mask out the owner's own slot (they can't capture themselves).
    one_hot = mx.zeros((G, N, P), dtype=mx.float32)
    one_hot_flat = one_hot.reshape(-1).at[self_idx].add(is_owned.reshape(-1))
    one_hot = one_hot_flat.reshape(G, N, P)
    damages_no_owner = (-forces) * (1.0 - one_hot)
    best_damage = damages_no_owner.max(axis=2)              # (G, N)
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

    # Frozen games: keep prior state.
    live_b = live_mask.reshape(G, 1)
    new_owner = mx.where(live_b, computed_owner, owner)
    new_strength_final = mx.where(live_b, computed_strength, strength)
    new_flow_valid = is_sending & live_b

    return new_owner, new_strength_final, new_flow_valid
