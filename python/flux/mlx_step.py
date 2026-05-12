"""MLX implementation of the flux step.

Tensor-only API. Caller is responsible for converting between (mx.array,
flow-array) representation and the GameState dataclass used elsewhere in the
package. No parity claim against the NumPy reference — MLX runs float32 with
its own reduction order. The browser only ever sees replay frames; matching
JS step bit-for-bit is no longer an invariant.

Single-game shape today; the leading dim (G,) gets added when the evolution
loop arrives. Where ops need batching later, the function is already written
to broadcast cleanly (one_hot, take_along_axis, .at[].add scatter).
"""
from __future__ import annotations

import mlx.core as mx

from .state import (
    ATTACK_BONUS,
    MAX_STRENGTH,
    MIN_STRENGTH_TO_SEND,
    REGEN_PER_SEC,
    TRANSFER_PER_SEC,
)


def step_mlx(
    owner: mx.array,      # (N,) int32, -1 = neutral
    strength: mx.array,   # (N,) float32, 0..MAX_STRENGTH
    flow_src: mx.array,   # (F,) int32
    flow_dst: mx.array,   # (F,) int32
    flow_player: mx.array,  # (F,) int32
    num_players: int,
    dt: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Returns (new_owner, new_strength, alive_mask).
    alive_mask: (F,) bool — flow's source still owned by player at start of step.
    Caller filters its Python-side flow list using this mask.
    """
    N = owner.shape[0]
    P = num_players
    F = flow_src.shape[0]

    is_owned = (owner != -1).astype(mx.float32)              # (N,)
    owner_idx = mx.maximum(owner, 0)                          # clamp -1 → 0 for one_hot
    # Build per-(cell, player) force accumulator (N, P).
    owner_oh = mx.zeros((N, P), dtype=mx.float32)
    owner_oh = owner_oh.at[mx.arange(N), owner_idx].add(is_owned)
    # Regen contribution.
    forces = REGEN_PER_SEC * dt * owner_oh                    # (N, P)

    if F > 0:
        flow_src_owner = owner[flow_src]                      # (F,)
        alive = flow_src_owner == flow_player                  # (F,) bool
        src_strength = strength[flow_src]
        can_send = src_strength >= MIN_STRENGTH_TO_SEND
        active = (alive & can_send).astype(mx.float32)        # (F,)
        k = TRANSFER_PER_SEC * dt

        # Outbound: forces[src, player] -= k * active
        out_delta = -k * active
        # Inbound: forces[dst, player] += k * active * (1 + ATTACK_BONUS * is_enemy)
        dst_owner = owner[flow_dst]
        is_enemy = (dst_owner != flow_player).astype(mx.float32)
        in_delta = k * active * (1.0 + ATTACK_BONUS * is_enemy)

        # Flatten + scatter-add. forces is (N, P) row-major.
        flat = forces.reshape(-1)
        flat = flat.at[flow_src * P + flow_player].add(out_delta)
        flat = flat.at[flow_dst * P + flow_player].add(in_delta)
        forces = flat.reshape(N, P)
    else:
        alive = mx.zeros((0,), dtype=mx.bool_)

    # Own-side force per cell (zero for neutral).
    own_force = mx.take_along_axis(forces, owner_idx.reshape(-1, 1), axis=1).reshape(-1)
    own_force = own_force * is_owned                          # (N,)
    total_force = forces.sum(axis=1)                          # (N,)
    opp_force = total_force - own_force                        # (N,)

    # For owned: strength += own - opp. For neutral: own_force=0 so strength -= opp.
    new_strength = strength + own_force - opp_force

    # Takeover: zero owner's slot then find best non-owner force.
    forces_excl_owner = forces - own_force.reshape(-1, 1) * owner_oh
    best_force = forces_excl_owner.max(axis=1)
    best_player = forces_excl_owner.argmax(axis=1).astype(owner.dtype)

    negative = new_strength < 0
    takeover = negative & (best_force > 0)
    new_owner = mx.where(takeover, best_player, owner)
    new_strength = mx.where(takeover, -new_strength, mx.maximum(new_strength, 0.0))
    new_strength = mx.minimum(new_strength, float(MAX_STRENGTH))

    return new_owner, new_strength, alive
