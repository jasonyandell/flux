"""v2 PPO + GNN actor-critic.

Same 2-layer GCN backbone as v1, but with the v2 13-action output space and
an extra input channel for the cell's current outflow count (so the policy
can sense its persistent decisions). All seats share parameters.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .state import K, MAX_EDGE, MAX_STRENGTH, NUM_ACTIONS, OPPOSITE_SLOT

# Direct neighbors only for v2 (K=6); but the GCN message-passing depth lets
# the network see further (r=2 with two MP layers).
NEIGHBOR_STRIDE = K
# Per-(seat, cell) input channels:
#   strength_norm, is_mine, is_enemy, is_neutral, is_dead, outflow_count_norm,
#   pressure_in_friendly_norm, pressure_in_enemy_norm, pressure_out_norm
IN_DIM = 9
HIDDEN = 32
POLICY_OUT = NUM_ACTIONS            # 13
VALUE_HIDDEN = 16


def _aggregate_neighbors(H: mx.array, neighbors: mx.array) -> mx.array:
    """Mean-aggregate H over each cell's direct hex neighbors. Empty slots
    (-1) do not contribute and don't count toward the divisor.
    H: (..., N, F).
    neighbors: (N, K) int32, -1 = no neighbor.
    """
    *lead, N, F = H.shape
    nb_ids = neighbors                                       # (N, K)
    nb_valid = nb_ids >= 0
    safe = mx.maximum(nb_ids, 0).reshape(-1)                 # (N*K,)
    gathered = mx.take(H, safe, axis=-2)                     # (*lead, N*K, F)
    gathered = gathered.reshape(*lead, N, K, F)
    mask_shape = [1] * len(lead) + [N, K, 1]
    mask = nb_valid.astype(mx.float32).reshape(mask_shape)
    summed = (gathered * mask).sum(axis=-2)
    count_shape = [1] * len(lead) + [N, 1]
    count = mx.maximum(
        nb_valid.astype(mx.float32).sum(axis=-1).reshape(count_shape),
        1.0,
    )
    return summed / count


def build_features(
    owner: mx.array,
    strength: mx.array,
    outflow: mx.array,
    edge_pressure: mx.array,
    neighbors: mx.array,
    num_players: int,
) -> mx.array:
    """Returns (G, S, N, IN_DIM) seat-relative features.

    Per-cell scalars (all 0..1 normalized):
      strength_norm        : cell strength
      is_mine/enemy/neutral/dead : ownership flags relative to the seat
      outflow_count_norm   : own active outflows / K
      pressure_in_friendly : flux arriving from cells owned by *this seat*
      pressure_in_enemy    : flux arriving from cells not owned by this seat
      pressure_out         : flux this cell is sending (only meaningful when
                             the cell belongs to this seat; zero otherwise)
    """
    G, N = owner.shape
    S = num_players
    out_shape = (G, S, N)
    seat_idx_2d = mx.arange(S).reshape(1, S, 1)
    seat_idx_3d = mx.arange(S).reshape(1, S, 1, 1)
    owner_b = owner.reshape(G, 1, N)

    is_dead_2d = (owner == -2)
    is_dead_b = is_dead_2d.reshape(G, 1, N)
    not_dead_b = mx.logical_not(is_dead_b)
    is_mine = mx.broadcast_to(
        ((owner_b == seat_idx_2d) & not_dead_b).astype(mx.float32), out_shape
    )
    is_neutral = mx.broadcast_to(
        ((owner_b == -1) & not_dead_b).astype(mx.float32), out_shape
    )
    is_enemy = mx.broadcast_to(
        ((owner_b != seat_idx_2d) & (owner_b >= 0)).astype(mx.float32), out_shape
    )
    is_dead = mx.broadcast_to(is_dead_b.astype(mx.float32), out_shape)
    strength_norm = mx.broadcast_to(
        (strength / MAX_STRENGTH).reshape(G, 1, N).astype(mx.float32), out_shape
    )
    outflow_count = outflow.astype(mx.float32).sum(axis=-1) / float(K)
    outflow_count_b = mx.broadcast_to(outflow_count.reshape(G, 1, N), out_shape)

    # ---- pressure features (seat-relative) ----
    # pressure_in_per_slot[g, c, k] = edge_pressure[g, neighbors[c, k], OPP[k]]
    nb_safe = mx.maximum(neighbors, 0)                            # (N, K)
    nb_valid = (neighbors >= 0).reshape(1, N, K)
    opp_mx = mx.array(OPPOSITE_SLOT.tolist(), dtype=mx.int32)
    flat_d_slot = (nb_safe * K + opp_mx.reshape(1, K)).reshape(-1)
    ep_flat = edge_pressure.reshape(G, N * K)
    pressure_in_per_slot = mx.take(ep_flat, flat_d_slot, axis=1).reshape(G, N, K)
    pressure_in_per_slot = pressure_in_per_slot * nb_valid.astype(mx.float32)

    # Per-seat split: pressure_in_friendly[g, s, c] = sum over k where
    # neighbors[c, k] is owned by seat s. pressure_in_enemy = total - friendly.
    owner_d = mx.take(owner, nb_safe.reshape(-1), axis=1).reshape(G, N, K)
    owner_d_b = owner_d.reshape(G, 1, N, K)
    is_friendly_per_seat = (owner_d_b == seat_idx_3d)                     # (G, S, N, K)
    pressure_in_per_slot_b = pressure_in_per_slot.reshape(G, 1, N, K)
    pressure_in_friendly = (
        pressure_in_per_slot_b * is_friendly_per_seat.astype(mx.float32)
    ).sum(axis=-1)                                                          # (G, S, N)
    pressure_in_total = pressure_in_per_slot.sum(axis=-1)                  # (G, N)
    pressure_in_enemy = pressure_in_total.reshape(G, 1, N) - pressure_in_friendly

    pressure_in_friendly_norm = (pressure_in_friendly / float(MAX_EDGE)).astype(mx.float32)
    pressure_in_enemy_norm = (pressure_in_enemy / float(MAX_EDGE)).astype(mx.float32)

    # pressure_out[g, s, c] = sum_k outflow[g, c, k] * edge_pressure[g, c, k]
    # but only when seat s owns c (else 0 — outflow's "pressure_out" is
    # meaningless for cells the seat doesn't control).
    own_pressure_sum = (outflow.astype(mx.float32) * edge_pressure).sum(axis=-1)  # (G, N)
    is_mine_b = (owner.reshape(G, 1, N) == seat_idx_2d).astype(mx.float32)        # (G, S, N)
    pressure_out_norm = (own_pressure_sum.reshape(G, 1, N) * is_mine_b) / float(MAX_EDGE)

    return mx.stack(
        [
            strength_norm, is_mine, is_enemy, is_neutral, is_dead,
            outflow_count_b,
            pressure_in_friendly_norm, pressure_in_enemy_norm, pressure_out_norm,
        ],
        axis=-1,
    )


class AttnActorCritic(nn.Module):
    """Attention-structured PPO policy. Same 3-layer GCN backbone as
    GNNActorCritic but the SET-action logits are produced by a 2-head
    soft-attention block mirroring the hand-designed `lightning_attn`
    solver:

        SET_k logit = (1 - α) · attack_q[k] + α · loop_q[k]

    where `attack_q[k]`, `loop_q[k]` are learned per-slot scores and α is a
    learned per-cell sigmoid scalar. CLEAR_k and NOOP keep generic linear
    heads so PPO can freely learn when to retract or stand pat.

    The attack/loop split is a *structural prior*, not a hard constraint:
    PPO can degenerate attack_q ≈ loop_q if the split isn't useful, in
    which case the policy reduces to a single SET-score head. The
    interesting question is whether the gradient finds a meaningful
    separation — that's the architectural test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w1_self = nn.Linear(IN_DIM, HIDDEN, bias=False)
        self.w1_neigh = nn.Linear(IN_DIM, HIDDEN, bias=True)
        self.w2_self = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w2_neigh = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.w3_self = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w3_neigh = nn.Linear(HIDDEN, HIDDEN, bias=True)
        # Per-slot attack and loop scores.
        self.attack_q_head = nn.Linear(HIDDEN, K)
        self.loop_q_head = nn.Linear(HIDDEN, K)
        # Per-cell mixing weight (sigmoid).
        self.alpha_head = nn.Linear(HIDDEN, 1)
        # Generic CLEAR and NOOP heads.
        self.clear_head = nn.Linear(HIDDEN, K)
        self.noop_head = nn.Linear(HIDDEN, 1)
        # Value head identical to GNNActorCritic.
        self.value_hidden = nn.Linear(HIDDEN, VALUE_HIDDEN)
        self.value_out = nn.Linear(VALUE_HIDDEN, 1)

    def forward(
        self,
        owner: mx.array,
        strength: mx.array,
        outflow: mx.array,
        edge_pressure: mx.array,
        neighbors: mx.array,
        num_players: int,
    ) -> tuple[mx.array, mx.array]:
        H0 = build_features(
            owner, strength, outflow, edge_pressure, neighbors, num_players,
        )
        H0_agg = _aggregate_neighbors(H0, neighbors)
        H1 = mx.maximum(self.w1_self(H0) + self.w1_neigh(H0_agg), 0)
        H1_agg = _aggregate_neighbors(H1, neighbors)
        H2 = mx.maximum(self.w2_self(H1) + self.w2_neigh(H1_agg), 0)
        H2_agg = _aggregate_neighbors(H2, neighbors)
        H3 = mx.maximum(self.w3_self(H2) + self.w3_neigh(H2_agg), 0)

        attack_q = self.attack_q_head(H3)            # (G, S, N, K)
        loop_q = self.loop_q_head(H3)                # (G, S, N, K)
        alpha = mx.sigmoid(self.alpha_head(H3))      # (G, S, N, 1)
        set_logits = (1.0 - alpha) * attack_q + alpha * loop_q  # (G, S, N, K)
        clear_logits = self.clear_head(H3)           # (G, S, N, K)
        noop_logit = self.noop_head(H3)              # (G, S, N, 1)
        policy_logits = mx.concatenate(
            [set_logits, clear_logits, noop_logit], axis=-1,
        )                                            # (G, S, N, 2K+1)

        v_per_cell = self.value_out(
            mx.maximum(self.value_hidden(H3), 0)
        ).squeeze(-1)
        G, N = owner.shape
        S = num_players
        seat_idx = mx.arange(S).reshape(1, S, 1)
        is_mine = (owner.reshape(G, 1, N) == seat_idx).astype(mx.float32)
        denom = mx.maximum(is_mine.sum(axis=-1), 1.0)
        value = (v_per_cell * is_mine).sum(axis=-1) / denom
        return policy_logits, value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class GNNActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 3-layer GCN: receptive field reaches 3 hops out, so a source cell
        # can see waste-prone terminators further down its chain.
        self.w1_self = nn.Linear(IN_DIM, HIDDEN, bias=False)
        self.w1_neigh = nn.Linear(IN_DIM, HIDDEN, bias=True)
        self.w2_self = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w2_neigh = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.w3_self = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w3_neigh = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.policy_head = nn.Linear(HIDDEN, POLICY_OUT)
        self.value_hidden = nn.Linear(HIDDEN, VALUE_HIDDEN)
        self.value_out = nn.Linear(VALUE_HIDDEN, 1)

    def forward(
        self,
        owner: mx.array,
        strength: mx.array,
        outflow: mx.array,
        edge_pressure: mx.array,
        neighbors: mx.array,
        num_players: int,
    ) -> tuple[mx.array, mx.array]:
        """Returns (policy_logits, value).
        policy_logits: (G, S, N, NUM_ACTIONS)
        value:         (G, S) — mean over seat-owned cells.
        """
        H0 = build_features(
            owner, strength, outflow, edge_pressure, neighbors, num_players,
        )
        H0_agg = _aggregate_neighbors(H0, neighbors)
        H1 = mx.maximum(self.w1_self(H0) + self.w1_neigh(H0_agg), 0)
        H1_agg = _aggregate_neighbors(H1, neighbors)
        H2 = mx.maximum(self.w2_self(H1) + self.w2_neigh(H1_agg), 0)
        H2_agg = _aggregate_neighbors(H2, neighbors)
        H3 = mx.maximum(self.w3_self(H2) + self.w3_neigh(H2_agg), 0)
        policy_logits = self.policy_head(H3)
        v_per_cell = self.value_out(
            mx.maximum(self.value_hidden(H3), 0)
        ).squeeze(-1)
        G, N = owner.shape
        S = num_players
        seat_idx = mx.arange(S).reshape(1, S, 1)
        is_mine = (owner.reshape(G, 1, N) == seat_idx).astype(mx.float32)
        denom = mx.maximum(is_mine.sum(axis=-1), 1.0)
        value = (v_per_cell * is_mine).sum(axis=-1) / denom
        return policy_logits, value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
