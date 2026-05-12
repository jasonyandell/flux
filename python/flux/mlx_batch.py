"""Batched MLX step + NN over G parallel games.

State layout for a batch of G games:
    owner       (G, N) int32       -1 = neutral
    strength    (G, N) float32     0..MAX_STRENGTH
    flow_src    (G, F_MAX) int32   flow source cell
    flow_dst    (G, F_MAX) int32   flow destination cell
    flow_player (G, F_MAX) int32   flow owner
    flow_valid  (G, F_MAX) bool    which flow slots are active
    live_mask   (G,) bool          which games are still simulating

NN weights tensor: (G, S, W) where S = num_seats = num_players. The shape
is identical per game so we can broadcast the matmul cleanly.

Aggressive (hand-coded) seats are NOT handled here — the batched NN still
runs for them (cheap) and the per-game flow reconcile in `game_loop.py`
overrides seat-0 (or whichever) with aggressive's chosen flows before the
next step.
"""
from __future__ import annotations

import mlx.core as mx

from .mlx_genome import (
    B1_OFF,
    B2_OFF,
    NEIGHBOR_STRIDE,
    NN_HIDDEN,
    NN_IN_DIM,
    NN_OUT_DIM,
    W2_OFF,
)
from .mlx_genome_v2 import (
    B1_OFF_V2,
    B2_OFF_V2,
    NN_HIDDEN_V2,
    NN_IN_DIM_V2,
    NN_OUT_DIM_V2,
    W2_OFF_V2,
)
from .mlx_genome_v3 import (
    NN_HIDDEN_V3,
    NN_IN_DIM_V3,
    NN_OUT_DIM_V3,
    OFF_B1 as V3_OFF_B1,
    OFF_B2 as V3_OFF_B2,
    OFF_B_OUT as V3_OFF_B_OUT,
    OFF_W1_NEIGH as V3_OFF_W1_NEIGH,
    OFF_W2_NEIGH as V3_OFF_W2_NEIGH,
    OFF_W2_SELF as V3_OFF_W2_SELF,
    OFF_W_OUT as V3_OFF_W_OUT,
    WEIGHTS_PER_GENOME_V3,
)
from .state import (
    ATTACK_BONUS,
    MAX_STRENGTH,
    MIN_STRENGTH_TO_SEND,
    REGEN_PER_SEC,
    TRANSFER_PER_SEC,
)
from .vision import STRIDE_V2

AGGRESSIVE_STRENGTH_THRESHOLD = 5.0


def step_batched(
    owner: mx.array,      # (G, N) int32
    strength: mx.array,   # (G, N) float32
    flow_src: mx.array,   # (G, F_MAX) int32
    flow_dst: mx.array,   # (G, F_MAX) int32
    flow_player: mx.array,  # (G, F_MAX) int32
    flow_valid: mx.array,  # (G, F_MAX) bool
    live_mask: mx.array,  # (G,) bool — done games are frozen
    num_players: int,
    dt: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """One step across G games in parallel.
    Returns (new_owner, new_strength, new_flow_valid).
    Frozen (~live) games come back unchanged.
    """
    G, N = owner.shape
    P = num_players
    live_f32 = live_mask.astype(mx.float32).reshape(G, 1)   # (G, 1) for broadcasting

    is_owned = (owner != -1).astype(mx.float32)              # (G, N)
    owner_safe = mx.maximum(owner, 0)                         # (G, N), clamp -1 → 0

    # ----- Forces: (G, N, P), flattened to (G*N*P,) for scatter-add -----
    forces_flat = mx.zeros(G * N * P, dtype=mx.float32)

    # Regen: forces[g, n, owner[g, n]] += REGEN * dt (for owned cells only).
    G_idx = mx.arange(G).reshape(G, 1)
    N_idx = mx.arange(N).reshape(1, N)
    regen_idx = (G_idx * (N * P) + N_idx * P + owner_safe).reshape(-1)   # (G*N,)
    regen_delta = (REGEN_PER_SEC * dt) * is_owned * live_f32             # (G, N)
    forces_flat = forces_flat.at[regen_idx].add(regen_delta.reshape(-1))

    # Flow contributions.
    G_idx_f = mx.arange(G).reshape(G, 1)
    flow_src_safe = mx.maximum(flow_src, 0)
    flow_dst_safe = mx.maximum(flow_dst, 0)
    flow_player_safe = mx.maximum(flow_player, 0)
    flat_idx_src = (G_idx_f * (N * P) + flow_src_safe * P + flow_player_safe).reshape(-1)
    flat_idx_dst = (G_idx_f * (N * P) + flow_dst_safe * P + flow_player_safe).reshape(-1)

    # Alive: flow.src.owner == flow.player (and the flow slot is valid).
    src_owner = mx.take_along_axis(owner, flow_src_safe, axis=1)         # (G, F_MAX)
    src_strength = mx.take_along_axis(strength, flow_src_safe, axis=1)
    alive = flow_valid & (src_owner == flow_player) & (src_strength >= MIN_STRENGTH_TO_SEND)
    alive_f32 = alive.astype(mx.float32)                                 # (G, F_MAX)

    dst_owner = mx.take_along_axis(owner, flow_dst_safe, axis=1)
    is_enemy = (dst_owner != flow_player).astype(mx.float32)             # (G, F_MAX)

    k = TRANSFER_PER_SEC * dt
    out_delta = -k * alive_f32 * live_f32                                # (G, F_MAX)
    in_delta = k * alive_f32 * (1.0 + ATTACK_BONUS * is_enemy) * live_f32

    forces_flat = forces_flat.at[flat_idx_src].add(out_delta.reshape(-1))
    forces_flat = forces_flat.at[flat_idx_dst].add(in_delta.reshape(-1))
    forces = forces_flat.reshape(G, N, P)

    # ----- Strength update -----
    own_force = mx.take_along_axis(forces, owner_safe.reshape(G, N, 1), axis=2).reshape(G, N)
    own_force = own_force * is_owned                                     # (G, N), zero for neutrals
    total_force = forces.sum(axis=2)                                     # (G, N)
    opp_force = total_force - own_force                                  # (G, N)

    new_strength = strength + own_force - opp_force

    # ----- Takeover detection -----
    # Zero out the owner's slot in forces so argmax picks a non-owner.
    one_hot = mx.zeros((G, N, P), dtype=mx.float32)
    one_hot_idx = regen_idx                                              # (G*N,)
    one_hot_flat = one_hot.reshape(-1).at[one_hot_idx].add(is_owned.reshape(-1))
    one_hot = one_hot_flat.reshape(G, N, P)
    forces_excl_owner = forces - own_force.reshape(G, N, 1) * one_hot
    best_force = forces_excl_owner.max(axis=2)                           # (G, N)
    best_player = forces_excl_owner.argmax(axis=2).astype(owner.dtype)   # (G, N)

    negative = new_strength < 0
    takeover = negative & (best_force > 0)
    computed_owner = mx.where(takeover, best_player, owner)
    computed_strength = mx.where(
        takeover,
        -new_strength,
        mx.maximum(new_strength, 0.0),
    )
    computed_strength = mx.minimum(computed_strength, float(MAX_STRENGTH))

    # Frozen games: keep prior state.
    live_b = live_mask.reshape(G, 1)
    new_owner = mx.where(live_b, computed_owner, owner)
    new_strength_final = mx.where(live_b, computed_strength, strength)
    new_flow_valid = flow_valid & alive & live_b

    return new_owner, new_strength_final, new_flow_valid


def build_flows_batched(
    weights: mx.array,       # (G, S, W) float32
    owner: mx.array,         # (G, N) int32
    strength: mx.array,      # (G, N) float32
    neighbors: mx.array,     # (N * STRIDE,) int32
    aggressive_seat: int,    # seat index running hand-coded aggressive (-1 = none)
    num_players: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Run batched NN forward + overlay the aggressive seat's hand-coded
    policy, then build dense (G, N) flow tensors from the chosen actions.

    Flow representation here is dense: one slot per cell. flow_src[g, n] = n.
    The slot is valid iff the cell is owned and the chosen action targets a
    real neighbor. This bypasses the Python toggle/cancel reconcile loop —
    flows refresh from scratch every AI tick.

    Returns (flow_src, flow_dst, flow_player, flow_valid), each (G, N).
    """
    G, N = owner.shape
    P = num_players
    STRIDE = NEIGHBOR_STRIDE

    # 1. NN argmax per (game, seat, cell).
    actions_all = nn_infer_batched(weights, owner, strength, neighbors, P)   # (G, S, N) int32

    # 2. For each cell, pick the action from its owner's NN row.
    owner_safe = mx.maximum(owner, 0)                                         # (G, N)
    nn_action = mx.take_along_axis(
        actions_all, owner_safe.reshape(G, 1, N), axis=1
    ).reshape(G, N)                                                           # (G, N)

    # 3. Aggressive overlay (if a seat is designated).
    if aggressive_seat >= 0:
        # For each cell, find the weakest non-friendly (and non-empty-slot)
        # neighbor relative to `aggressive_seat`. Vectorized over (G, N).
        nb_ids_2d = neighbors.reshape(N, STRIDE)                              # (N, STRIDE)
        nb_valid_2d = nb_ids_2d >= 0                                           # (N, STRIDE)
        safe_ids_flat = mx.maximum(nb_ids_2d, 0).reshape(-1)                   # (N*STRIDE,)
        nb_strength = mx.take(strength, safe_ids_flat, axis=1).reshape(G, N, STRIDE)
        nb_owner = mx.take(owner, safe_ids_flat, axis=1).reshape(G, N, STRIDE)
        # Masked: friendly OR invalid slot → mark with a huge sentinel so argmin avoids it.
        nb_valid_b = nb_valid_2d.reshape(1, N, STRIDE)
        is_friend_or_invalid = (nb_owner == aggressive_seat) | ~nb_valid_b
        BIG = 1.0e9
        masked = mx.where(is_friend_or_invalid, mx.full(nb_strength.shape, BIG, dtype=mx.float32), nb_strength)
        aggressive_action = masked.argmin(axis=-1).astype(mx.int32)            # (G, N)
        agg_min_strength = masked.min(axis=-1)                                  # (G, N)
        has_target = agg_min_strength < BIG / 2.0                              # (G, N) bool

        agg_owned = owner == aggressive_seat                                    # (G, N)
        agg_strong = strength > AGGRESSIVE_STRENGTH_THRESHOLD                   # (G, N)
        agg_active = agg_owned & agg_strong & has_target                        # (G, N)
        # Final per-cell action: aggressive's choice where active, else NN.
        chosen_action = mx.where(agg_active, aggressive_action, nn_action)
        # Aggressive-owned cells that aren't "active" must NOT emit a flow
        # (they're below threshold or have no target). Force them to noop (18).
        agg_silent = agg_owned & ~agg_active
        chosen_action = mx.where(
            agg_silent,
            mx.full(chosen_action.shape, 18, dtype=chosen_action.dtype),
            chosen_action,
        )
    else:
        chosen_action = nn_action

    # 4. Build flow tensors. Dense: src=cell index, dst=neighbors[cell, chosen_action].
    action_in_range = (chosen_action >= 0) & (chosen_action < STRIDE)          # (G, N)
    chosen_safe = mx.maximum(mx.minimum(chosen_action, STRIDE - 1), 0)         # (G, N)
    # Flat neighbor lookup: flat_idx[g, n] = n * STRIDE + chosen_safe[g, n].
    n_arange = mx.arange(N).reshape(1, N) * STRIDE                              # (1, N)
    flat_idx = (n_arange + chosen_safe).reshape(-1)                             # (G * N,)
    flow_dst_flat = mx.take(neighbors, flat_idx, axis=0)                        # (G*N,)
    flow_dst = flow_dst_flat.reshape(G, N)
    flow_src = mx.broadcast_to(mx.arange(N).reshape(1, N), (G, N)).astype(mx.int32)
    flow_player = mx.where(owner >= 0, owner, mx.full(owner.shape, -1, dtype=owner.dtype))
    flow_valid = (owner >= 0) & action_in_range & (flow_dst >= 0)
    return flow_src.astype(mx.int32), flow_dst.astype(mx.int32), flow_player.astype(mx.int32), flow_valid


def nn_infer_batched_v2(
    weights: mx.array,           # (G, S, W_V2) float32, W_V2 = 6451
    owner: mx.array,             # (G, N) int32
    strength: mx.array,          # (G, N) float32
    vision: mx.array,            # (N * STRIDE_V2,) int32 — 3-hop neighbor table
    num_players: int,
) -> mx.array:
    """V2 NN forward. 181 inputs (own + 36×5), same hidden+output as v1."""
    G, N = owner.shape
    S = num_players
    STRIDE = STRIDE_V2

    nb_ids = vision.reshape(N, STRIDE)
    nb_valid_2d = nb_ids >= 0
    safe_ids = mx.maximum(nb_ids, 0)
    safe_flat = safe_ids.reshape(-1)
    nb_strength = mx.take(strength, safe_flat, axis=1).reshape(G, N, STRIDE)
    nb_owner = mx.take(owner, safe_flat, axis=1).reshape(G, N, STRIDE)
    nb_valid_b = nb_valid_2d.reshape(1, N, STRIDE)

    nb_strength_norm = mx.where(
        nb_valid_b, nb_strength / MAX_STRENGTH, mx.zeros_like(nb_strength)
    )

    seat_idx = mx.arange(S).reshape(1, S, 1, 1)
    nb_owner_b = nb_owner.reshape(G, 1, N, STRIDE)
    nb_valid_b4 = nb_valid_2d.reshape(1, 1, N, STRIDE)
    is_friend = (nb_owner_b == seat_idx) & nb_valid_b4
    is_enemy = (nb_owner_b != seat_idx) & nb_valid_b4 & (nb_owner_b != -1)
    is_neutral = ~nb_valid_b4 | (nb_valid_b4 & (nb_owner_b == -1))

    out_shape = (G, S, N, STRIDE)
    fa = mx.broadcast_to(is_friend.astype(mx.float32), out_shape)
    en = mx.broadcast_to(is_enemy.astype(mx.float32), out_shape)
    nu = mx.broadcast_to(is_neutral.astype(mx.float32), out_shape)
    nb_strength_full = mx.broadcast_to(
        nb_strength_norm.reshape(G, 1, N, STRIDE).astype(mx.float32), out_shape
    )

    nb_features = mx.stack([nb_strength_full, fa, fa, en, nu], axis=-1)
    nb_features = nb_features.reshape(G, S, N, STRIDE * 5)
    self_feat = (strength / MAX_STRENGTH).reshape(G, 1, N, 1)
    self_feat_full = mx.broadcast_to(self_feat, (G, S, N, 1)).astype(mx.float32)
    features = mx.concatenate([self_feat_full, nb_features], axis=-1)   # (G, S, N, 181)

    W1 = weights[..., :B1_OFF_V2].reshape(G, S, NN_IN_DIM_V2, NN_HIDDEN_V2)
    b1 = weights[..., B1_OFF_V2:W2_OFF_V2].reshape(G, S, 1, NN_HIDDEN_V2)
    W2 = weights[..., W2_OFF_V2:B2_OFF_V2].reshape(G, S, NN_HIDDEN_V2, NN_OUT_DIM_V2)
    b2 = weights[..., B2_OFF_V2:].reshape(G, S, 1, NN_OUT_DIM_V2)

    h = mx.tanh(features @ W1 + b1)
    out = h @ W2 + b2                                                    # (G, S, N, 19)
    return out.argmax(axis=-1).astype(mx.int32)                          # (G, S, N)


def build_flows_batched_v2(
    weights: mx.array,           # (G, S, W_V2)
    owner: mx.array,
    strength: mx.array,
    graph_neighbors: mx.array,   # (N * 18,) — for action → flow dst lookup
    vision: mx.array,            # (N * 36,) — for NN feature build
    aggressive_seat: int,
    num_players: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """V2 flow build. NN uses `vision` (3-hop); aggressive overlay and the
    final action→dst lookup use `graph_neighbors` (1- and 2-hop, since flows
    can only travel via game edges)."""
    G, N = owner.shape
    P = num_players
    STRIDE_G = NEIGHBOR_STRIDE  # graph stride = 18

    actions_all = nn_infer_batched_v2(weights, owner, strength, vision, P)   # (G, S, N) int32
    owner_safe = mx.maximum(owner, 0)
    nn_action = mx.take_along_axis(
        actions_all, owner_safe.reshape(G, 1, N), axis=1
    ).reshape(G, N)

    if aggressive_seat >= 0:
        nb_ids_2d = graph_neighbors.reshape(N, STRIDE_G)
        nb_valid_2d = nb_ids_2d >= 0
        safe_ids_flat = mx.maximum(nb_ids_2d, 0).reshape(-1)
        nb_strength = mx.take(strength, safe_ids_flat, axis=1).reshape(G, N, STRIDE_G)
        nb_owner = mx.take(owner, safe_ids_flat, axis=1).reshape(G, N, STRIDE_G)
        nb_valid_b = nb_valid_2d.reshape(1, N, STRIDE_G)
        is_friend_or_invalid = (nb_owner == aggressive_seat) | ~nb_valid_b
        BIG = 1.0e9
        masked = mx.where(is_friend_or_invalid, mx.full(nb_strength.shape, BIG, dtype=mx.float32), nb_strength)
        aggressive_action = masked.argmin(axis=-1).astype(mx.int32)
        agg_min_strength = masked.min(axis=-1)
        has_target = agg_min_strength < BIG / 2.0
        agg_owned = owner == aggressive_seat
        agg_strong = strength > AGGRESSIVE_STRENGTH_THRESHOLD
        agg_active = agg_owned & agg_strong & has_target
        chosen_action = mx.where(agg_active, aggressive_action, nn_action)
        agg_silent = agg_owned & ~agg_active
        chosen_action = mx.where(
            agg_silent,
            mx.full(chosen_action.shape, 18, dtype=chosen_action.dtype),
            chosen_action,
        )
    else:
        chosen_action = nn_action

    action_in_range = (chosen_action >= 0) & (chosen_action < STRIDE_G)
    chosen_safe = mx.maximum(mx.minimum(chosen_action, STRIDE_G - 1), 0)
    n_arange = mx.arange(N).reshape(1, N) * STRIDE_G
    flat_idx = (n_arange + chosen_safe).reshape(-1)
    flow_dst_flat = mx.take(graph_neighbors, flat_idx, axis=0)
    flow_dst = flow_dst_flat.reshape(G, N)
    flow_src = mx.broadcast_to(mx.arange(N).reshape(1, N), (G, N)).astype(mx.int32)
    flow_player = mx.where(owner >= 0, owner, mx.full(owner.shape, -1, dtype=owner.dtype))
    flow_valid = (owner >= 0) & action_in_range & (flow_dst >= 0)
    return flow_src.astype(mx.int32), flow_dst.astype(mx.int32), flow_player.astype(mx.int32), flow_valid


def nn_infer_batched(
    weights: mx.array,      # (G, S, W) float32 — one genome per (game, seat)
    owner: mx.array,        # (G, N) int32
    strength: mx.array,     # (G, N) float32
    neighbors: mx.array,    # (N * NEIGHBOR_STRIDE,) int32 — shared across games
    num_players: int,
) -> mx.array:
    """Returns (G, S, N) int32 — argmax action per (game, seat, cell)."""
    G, N = owner.shape
    S = num_players
    STRIDE = NEIGHBOR_STRIDE

    nb_ids = neighbors.reshape(N, STRIDE)                                # (N, STRIDE)
    nb_valid_2d = nb_ids >= 0                                            # (N, STRIDE)
    safe_ids = mx.maximum(nb_ids, 0)                                     # (N, STRIDE)

    # Gather neighbor strength and owner per game. strength[:, safe_ids]:
    # MLX supports gather along an axis via take, but multi-axis fancy indexing
    # is restricted. Use take with reshaped indices.
    safe_flat = safe_ids.reshape(-1)                                     # (N*STRIDE,)
    nb_strength = mx.take(strength, safe_flat, axis=1).reshape(G, N, STRIDE)
    nb_owner = mx.take(owner, safe_flat, axis=1).reshape(G, N, STRIDE)
    nb_valid_b = nb_valid_2d.reshape(1, N, STRIDE)                       # broadcast over G

    nb_strength_norm = mx.where(
        nb_valid_b, nb_strength / MAX_STRENGTH, mx.zeros_like(nb_strength)
    )                                                                    # (G, N, STRIDE)

    # Per-seat masks. Want (G, S, N, STRIDE).
    seat_idx = mx.arange(S).reshape(1, S, 1, 1)
    nb_owner_b = nb_owner.reshape(G, 1, N, STRIDE)
    nb_valid_b4 = nb_valid_2d.reshape(1, 1, N, STRIDE)
    is_friend = (nb_owner_b == seat_idx) & nb_valid_b4
    is_enemy = (nb_owner_b != seat_idx) & nb_valid_b4 & (nb_owner_b != -1)
    is_neutral = ~nb_valid_b4 | (nb_valid_b4 & (nb_owner_b == -1))

    out_shape = (G, S, N, STRIDE)
    fa = mx.broadcast_to(is_friend.astype(mx.float32), out_shape)
    # NumPy reference duplicates the friend channel.
    en = mx.broadcast_to(is_enemy.astype(mx.float32), out_shape)
    nu = mx.broadcast_to(is_neutral.astype(mx.float32), out_shape)

    nb_strength_full = mx.broadcast_to(
        nb_strength_norm.reshape(G, 1, N, STRIDE).astype(mx.float32), out_shape
    )

    # Stack along a new last axis then flatten the (STRIDE, 5) block.
    nb_features = mx.stack([nb_strength_full, fa, fa, en, nu], axis=-1)  # (G, S, N, STRIDE, 5)
    nb_features = nb_features.reshape(G, S, N, STRIDE * 5)

    self_feat = (strength / MAX_STRENGTH).reshape(G, 1, N, 1)
    self_feat_full = mx.broadcast_to(self_feat, (G, S, N, 1)).astype(mx.float32)
    features = mx.concatenate([self_feat_full, nb_features], axis=-1)    # (G, S, N, 91)

    # Unpack weights into per-(game, seat) matrices.
    W1 = weights[..., :B1_OFF].reshape(G, S, NN_IN_DIM, NN_HIDDEN)
    b1 = weights[..., B1_OFF:W2_OFF].reshape(G, S, 1, NN_HIDDEN)
    W2 = weights[..., W2_OFF:B2_OFF].reshape(G, S, NN_HIDDEN, NN_OUT_DIM)
    b2 = weights[..., B2_OFF:].reshape(G, S, 1, NN_OUT_DIM)

    h = mx.tanh(features @ W1 + b1)                                      # (G, S, N, 32)
    out = h @ W2 + b2                                                    # (G, S, N, 19)
    return out.argmax(axis=-1).astype(mx.int32)                          # (G, S, N)


# ===== v3: 2-layer GCN-style message-passing policy =====


def _graph_neighbor_aggregate(
    H: mx.array,                  # (G, S, N, F)
    graph_neighbors: mx.array,    # (N * NEIGHBOR_STRIDE,) int32, -1 = empty
) -> mx.array:
    """Mean-aggregate H over each cell's graph neighbors. Empty slots (-1)
    don't contribute and don't count toward the divisor.
    Returns (G, S, N, F)."""
    G, S, N, F = H.shape
    STRIDE = NEIGHBOR_STRIDE
    nb_ids = graph_neighbors.reshape(N, STRIDE)
    nb_valid_2d = nb_ids >= 0
    safe_flat = mx.maximum(nb_ids, 0).reshape(-1)            # (N*STRIDE,)
    gathered = mx.take(H, safe_flat, axis=2).reshape(G, S, N, STRIDE, F)
    mask = nb_valid_2d.astype(mx.float32).reshape(1, 1, N, STRIDE, 1)
    gathered = gathered * mask
    summed = gathered.sum(axis=3)                            # (G, S, N, F)
    count = nb_valid_2d.astype(mx.float32).sum(axis=-1).reshape(1, 1, N, 1)
    count = mx.maximum(count, 1.0)
    return summed / count


def _build_features_v3(
    owner: mx.array,        # (G, N) int32
    strength: mx.array,     # (G, N) float32
    num_players: int,
) -> mx.array:
    """Per-(game, seat, cell) feature tensor for v3.
    Channels: [strength_norm, is_mine, is_enemy, is_neutral_or_empty].
    Returns (G, S, N, 4)."""
    G, N = owner.shape
    S = num_players
    out_shape = (G, S, N)
    seat_idx = mx.arange(S).reshape(1, S, 1)
    owner_b = owner.reshape(G, 1, N)
    is_mine = mx.broadcast_to((owner_b == seat_idx).astype(mx.float32), out_shape)
    is_neutral = mx.broadcast_to((owner_b == -1).astype(mx.float32), out_shape)
    is_enemy = mx.broadcast_to(
        ((owner_b != seat_idx) & (owner_b != -1)).astype(mx.float32), out_shape
    )
    strength_norm = mx.broadcast_to(
        (strength / MAX_STRENGTH).reshape(G, 1, N).astype(mx.float32), out_shape
    )
    return mx.stack([strength_norm, is_mine, is_enemy, is_neutral], axis=-1)


def nn_infer_batched_v3(
    weights: mx.array,           # (G, S, WEIGHTS_PER_GENOME_V3)
    owner: mx.array,             # (G, N) int32
    strength: mx.array,          # (G, N) float32
    graph_neighbors: mx.array,   # (N * NEIGHBOR_STRIDE,) int32
    num_players: int,
) -> mx.array:
    """V3 GNN forward. Returns (G, S, N) int32 argmax actions."""
    G, N = owner.shape
    S = num_players
    F_in = NN_IN_DIM_V3
    F_h = NN_HIDDEN_V3
    F_out = NN_OUT_DIM_V3

    W1_self = weights[..., :V3_OFF_W1_NEIGH].reshape(G, S, F_in, F_h)
    W1_neigh = weights[..., V3_OFF_W1_NEIGH:V3_OFF_B1].reshape(G, S, F_in, F_h)
    b1 = weights[..., V3_OFF_B1:V3_OFF_W2_SELF].reshape(G, S, 1, F_h)
    W2_self = weights[..., V3_OFF_W2_SELF:V3_OFF_W2_NEIGH].reshape(G, S, F_h, F_h)
    W2_neigh = weights[..., V3_OFF_W2_NEIGH:V3_OFF_B2].reshape(G, S, F_h, F_h)
    b2 = weights[..., V3_OFF_B2:V3_OFF_W_OUT].reshape(G, S, 1, F_h)
    W_out = weights[..., V3_OFF_W_OUT:V3_OFF_B_OUT].reshape(G, S, F_h, F_out)
    b_out = weights[..., V3_OFF_B_OUT:].reshape(G, S, 1, F_out)

    H0 = _build_features_v3(owner, strength, S)             # (G, S, N, 4)
    H0_agg = _graph_neighbor_aggregate(H0, graph_neighbors)
    H1 = mx.maximum(H0 @ W1_self + H0_agg @ W1_neigh + b1, 0)
    H1_agg = _graph_neighbor_aggregate(H1, graph_neighbors)
    H2 = mx.maximum(H1 @ W2_self + H1_agg @ W2_neigh + b2, 0)
    out = H2 @ W_out + b_out                                 # (G, S, N, 19)
    return out.argmax(axis=-1).astype(mx.int32)


def build_flows_batched_v3(
    weights: mx.array,           # (G, S, WEIGHTS_PER_GENOME_V3)
    owner: mx.array,
    strength: mx.array,
    graph_neighbors: mx.array,   # (N * 18,) — used for both NN message passing
                                  # and action → flow dst lookup
    aggressive_seat: int,
    num_players: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """V3 flow build. GNN runs message passing over the same graph that flows
    use; aggressive overlay + action→dst use the same neighbor table."""
    G, N = owner.shape
    P = num_players
    STRIDE_G = NEIGHBOR_STRIDE

    actions_all = nn_infer_batched_v3(weights, owner, strength, graph_neighbors, P)
    owner_safe = mx.maximum(owner, 0)
    nn_action = mx.take_along_axis(
        actions_all, owner_safe.reshape(G, 1, N), axis=1
    ).reshape(G, N)

    if aggressive_seat >= 0:
        nb_ids_2d = graph_neighbors.reshape(N, STRIDE_G)
        nb_valid_2d = nb_ids_2d >= 0
        safe_ids_flat = mx.maximum(nb_ids_2d, 0).reshape(-1)
        nb_strength = mx.take(strength, safe_ids_flat, axis=1).reshape(G, N, STRIDE_G)
        nb_owner = mx.take(owner, safe_ids_flat, axis=1).reshape(G, N, STRIDE_G)
        nb_valid_b = nb_valid_2d.reshape(1, N, STRIDE_G)
        is_friend_or_invalid = (nb_owner == aggressive_seat) | ~nb_valid_b
        BIG = 1.0e9
        masked = mx.where(
            is_friend_or_invalid,
            mx.full(nb_strength.shape, BIG, dtype=mx.float32),
            nb_strength,
        )
        aggressive_action = masked.argmin(axis=-1).astype(mx.int32)
        agg_min_strength = masked.min(axis=-1)
        has_target = agg_min_strength < BIG / 2.0
        agg_owned = owner == aggressive_seat
        agg_strong = strength > AGGRESSIVE_STRENGTH_THRESHOLD
        agg_active = agg_owned & agg_strong & has_target
        chosen_action = mx.where(agg_active, aggressive_action, nn_action)
        agg_silent = agg_owned & ~agg_active
        chosen_action = mx.where(
            agg_silent,
            mx.full(chosen_action.shape, 18, dtype=chosen_action.dtype),
            chosen_action,
        )
    else:
        chosen_action = nn_action

    action_in_range = (chosen_action >= 0) & (chosen_action < STRIDE_G)
    chosen_safe = mx.maximum(mx.minimum(chosen_action, STRIDE_G - 1), 0)
    n_arange = mx.arange(N).reshape(1, N) * STRIDE_G
    flat_idx = (n_arange + chosen_safe).reshape(-1)
    flow_dst_flat = mx.take(graph_neighbors, flat_idx, axis=0)
    flow_dst = flow_dst_flat.reshape(G, N)
    flow_src = mx.broadcast_to(mx.arange(N).reshape(1, N), (G, N)).astype(mx.int32)
    flow_player = mx.where(owner >= 0, owner, mx.full(owner.shape, -1, dtype=owner.dtype))
    flow_valid = (owner >= 0) & action_in_range & (flow_dst >= 0)
    return (
        flow_src.astype(mx.int32),
        flow_dst.astype(mx.int32),
        flow_player.astype(mx.int32),
        flow_valid,
    )


def build_flows_from_actions(
    actions_all: mx.array,       # (G, S, N) int32 — chosen action per (game, seat, cell)
    owner: mx.array,             # (G, N) int32
    graph_neighbors: mx.array,   # (N * NEIGHBOR_STRIDE,) int32
    dead_mask: mx.array | None = None,  # (G, N) bool — flows landing on dead cells get invalidated
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Build dense (G, N) flow tensors from per-(seat, cell) action choices.

    The per-cell action used is the one chosen by the cell's *owner* seat
    (taken from actions_all along the S axis). Cells without an owner emit no
    flow; cells where the chosen action is out of range (≥ 18 = noop) emit no
    flow either.

    This is the rollout-mode counterpart to `build_flows_batched_v3` — that
    function runs the policy itself; this one accepts externally-supplied
    actions (e.g. sampled from a categorical for PPO).
    """
    G, N = owner.shape
    STRIDE_G = NEIGHBOR_STRIDE
    owner_safe = mx.maximum(owner, 0)
    chosen_action = mx.take_along_axis(
        actions_all, owner_safe.reshape(G, 1, N), axis=1
    ).reshape(G, N)
    action_in_range = (chosen_action >= 0) & (chosen_action < STRIDE_G)
    chosen_safe = mx.maximum(mx.minimum(chosen_action, STRIDE_G - 1), 0)
    n_arange = mx.arange(N).reshape(1, N) * STRIDE_G
    flat_idx = (n_arange + chosen_safe).reshape(-1)
    flow_dst_flat = mx.take(graph_neighbors, flat_idx, axis=0)
    flow_dst = flow_dst_flat.reshape(G, N)
    flow_src = mx.broadcast_to(mx.arange(N).reshape(1, N), (G, N)).astype(mx.int32)
    flow_player = mx.where(owner >= 0, owner, mx.full(owner.shape, -1, dtype=owner.dtype))
    flow_valid = (owner >= 0) & action_in_range & (flow_dst >= 0)
    if dead_mask is not None:
        # Drop flows that land on dead cells. Source-side dead is mostly
        # redundant (dead cells have owner=-1 already) — kept for safety.
        flow_dst_safe = mx.maximum(flow_dst, 0)
        dst_is_dead = mx.take_along_axis(dead_mask, flow_dst_safe, axis=1)
        flow_valid = flow_valid & mx.logical_not(dst_is_dead) & mx.logical_not(dead_mask)
    return (
        flow_src.astype(mx.int32),
        flow_dst.astype(mx.int32),
        flow_player.astype(mx.int32),
        flow_valid,
    )
