"""v2 PPO + GNN actor-critic.

Same 2-layer GCN backbone as v1, but with the v2 13-action output space and
an extra input channel for the cell's current outflow count (so the policy
can sense its persistent decisions). All seats share parameters.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

import numpy as np

from .edge_features import (
    EDGE_BLOCKED,
    EDGE_CATEGORY_INDEX,
    EDGE_CATEGORY_NAMES,
    EDGE_FEATURE_INDEX,
    EDGE_MINE_TO_ENEMY,
    EDGE_MINE_TO_FRIENDLY_FILL,
    EDGE_MINE_TO_FRIENDLY_RELAY,
    EDGE_MINE_TO_FRIENDLY_SINK,
    EDGE_MINE_TO_NEUTRAL,
    NUM_EDGE_CATEGORIES,
    build_edge_features,
)
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
EDGE_FEATURE_DIM = 25
EDGE_HIDDEN = 32
EDGE_TYPE_NAMES = EDGE_CATEGORY_NAMES
EDGE_NUM_TYPES = NUM_EDGE_CATEGORIES
EDGE_CHANNEL_NAMES = (
    "attack_flow",
    "expand_flow",
    "relay_flow",
    "fill_flow",
    "sink_risk",
    "threat_in",
    "stored_pressure",
    "front_break",
    "follow_through",
)
EDGE_NUM_CHANNELS = len(EDGE_CHANNEL_NAMES)
EDGE_TYPE_BLOCKED = EDGE_BLOCKED


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


def build_edge_auxiliary_targets_np(
    owner: np.ndarray,
    strength: np.ndarray,
    outflow: np.ndarray,
    edge_pressure: np.ndarray,
    neighbors: np.ndarray,
    num_players: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive edge-type labels and channel targets from v2 state.

    Returns:
      edge_type: (G, S, N, K) int labels in EDGE_TYPE_NAMES order.
      channels:  (G, S, N, K, EDGE_NUM_CHANNELS) float targets.
      mask:      (G, S, N, K) true for valid non-dead source/destination edges.

    Category labels come from ``edge_features.build_edge_features`` so the
    policy, pretraining script, heuristic, and trainer metrics share one
    representation. Channel targets stay deliberately simple and soft: they
    teach local edge vocabulary without hard-coding pulse timing.
    """
    batch = build_edge_features(
        owner=owner,
        strength=strength,
        outflow=outflow,
        edge_pressure=edge_pressure,
        neighbors=neighbors,
        num_players=num_players,
    )
    labels = batch.category.astype(np.int32)
    channels = np.zeros(
        labels.shape + (EDGE_NUM_CHANNELS,),
        dtype=np.float32,
    )
    mine_to_enemy = labels == EDGE_MINE_TO_ENEMY
    mine_to_neutral = labels == EDGE_MINE_TO_NEUTRAL
    mine_to_friendly_relay = labels == EDGE_MINE_TO_FRIENDLY_RELAY
    mine_to_friendly_sink = labels == EDGE_MINE_TO_FRIENDLY_SINK
    mine_to_friendly_fill = labels == EDGE_MINE_TO_FRIENDLY_FILL
    enemy_to_mine = labels == EDGE_CATEGORY_INDEX["enemy_to_mine"]

    f = batch.features
    source_strength = f[..., EDGE_FEATURE_INDEX["source_strength"]]
    dest_strength = f[..., EDGE_FEATURE_INDEX["dest_strength"]]
    dest_headroom = f[..., EDGE_FEATURE_INDEX["dest_headroom"]]
    edge_pressure_norm = f[..., EDGE_FEATURE_INDEX["edge_pressure"]]
    active = f[..., EDGE_FEATURE_INDEX["outflow_active"]]

    channels[..., EDGE_CHANNEL_NAMES.index("attack_flow")] = mine_to_enemy
    channels[..., EDGE_CHANNEL_NAMES.index("expand_flow")] = mine_to_neutral
    channels[..., EDGE_CHANNEL_NAMES.index("relay_flow")] = mine_to_friendly_relay
    channels[..., EDGE_CHANNEL_NAMES.index("fill_flow")] = mine_to_friendly_fill
    channels[..., EDGE_CHANNEL_NAMES.index("sink_risk")] = mine_to_friendly_sink
    channels[..., EDGE_CHANNEL_NAMES.index("threat_in")] = enemy_to_mine
    channels[..., EDGE_CHANNEL_NAMES.index("stored_pressure")] = edge_pressure_norm
    channels[..., EDGE_CHANNEL_NAMES.index("front_break")] = (
        mine_to_enemy.astype(np.float32) * dest_headroom
    )
    channels[..., EDGE_CHANNEL_NAMES.index("follow_through")] = (
        mine_to_enemy.astype(np.float32)
        * active
        * source_strength
        * (dest_strength < 0.5).astype(np.float32)
    )
    mask = batch.masks.category_known.copy()
    channels *= mask[..., None].astype(np.float32)
    return labels, channels, mask


class EdgeAwareActorCritic(nn.Module):
    """Edge-scoring actor-critic with the same public forward contract.

    The baseline `GNNActorCritic` stays node-centric. This variant reuses the
    three-hop node encoder, scores each source-owned directed edge from
    source embedding + destination embedding + per-edge features, then maps
    open/clear preferences back into the v2 13-action surface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w1_self = nn.Linear(IN_DIM, HIDDEN, bias=False)
        self.w1_neigh = nn.Linear(IN_DIM, HIDDEN, bias=True)
        self.w2_self = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w2_neigh = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.w3_self = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w3_neigh = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.edge_feature = nn.Linear(EDGE_FEATURE_DIM, EDGE_HIDDEN)
        self.edge_pair = nn.Linear(HIDDEN * 2 + EDGE_HIDDEN, EDGE_HIDDEN)
        self.edge_score = nn.Linear(EDGE_HIDDEN, 2)
        self.noop_head = nn.Linear(HIDDEN, 1)
        self.edge_type_head = nn.Linear(EDGE_HIDDEN, EDGE_NUM_TYPES)
        self.edge_channel_head = nn.Linear(EDGE_HIDDEN, EDGE_NUM_CHANNELS)
        self.value_hidden = nn.Linear(HIDDEN, VALUE_HIDDEN)
        self.value_out = nn.Linear(VALUE_HIDDEN, 1)

    def _encode_nodes(
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
        return H0, H3

    def _edge_features(
        self,
        H0: mx.array,
        outflow: mx.array,
        edge_pressure: mx.array,
        neighbors: mx.array,
    ) -> mx.array:
        G, S, N, _ = H0.shape
        nb_safe = mx.maximum(neighbors, 0)
        nb_valid = (neighbors >= 0).astype(mx.float32)
        H0_dst = mx.take(H0, nb_safe.reshape(-1), axis=-2).reshape(G, S, N, K, IN_DIM)
        H0_src = H0.reshape(G, S, N, 1, IN_DIM)
        outflow_f = mx.broadcast_to(
            outflow.astype(mx.float32).reshape(G, 1, N, K), (G, S, N, K)
        )
        edge_pressure_norm = mx.broadcast_to(
            (edge_pressure / float(MAX_EDGE)).reshape(G, 1, N, K), (G, S, N, K)
        )
        valid = mx.broadcast_to(nb_valid.reshape(1, 1, N, K), (G, S, N, K))
        slot_onehot = mx.broadcast_to(
            mx.eye(K, dtype=mx.float32).reshape(1, 1, 1, K, K),
            (G, S, N, K, K),
        )
        scalars = mx.stack(
            [
                valid,
                mx.broadcast_to(H0_src[..., 0], (G, S, N, K)),
                H0_dst[..., 0],
                1.0 - mx.broadcast_to(H0_src[..., 0], (G, S, N, K)),
                1.0 - H0_dst[..., 0],
                outflow_f,
                edge_pressure_norm,
                mx.broadcast_to(H0_src[..., 6], (G, S, N, K)),
                mx.broadcast_to(H0_src[..., 7], (G, S, N, K)),
                H0_dst[..., 6],
                H0_dst[..., 7],
                H0_dst[..., 5],
                H0_dst[..., 1],
                H0_dst[..., 2],
                H0_dst[..., 3],
                H0_dst[..., 4],
                mx.broadcast_to(H0_src[..., 1], (G, S, N, K)),
                mx.broadcast_to(H0_src[..., 2], (G, S, N, K)),
                mx.broadcast_to(H0_src[..., 3], (G, S, N, K)),
            ],
            axis=-1,
        )
        return mx.concatenate([scalars, slot_onehot], axis=-1)

    def _edge_hidden(
        self,
        H0: mx.array,
        H: mx.array,
        outflow: mx.array,
        edge_pressure: mx.array,
        neighbors: mx.array,
    ) -> mx.array:
        G, S, N, _ = H.shape
        nb_safe = mx.maximum(neighbors, 0)
        H_dst = mx.take(H, nb_safe.reshape(-1), axis=-2).reshape(G, S, N, K, HIDDEN)
        H_src = mx.broadcast_to(H.reshape(G, S, N, 1, HIDDEN), (G, S, N, K, HIDDEN))
        E = mx.maximum(self.edge_feature(self._edge_features(H0, outflow, edge_pressure, neighbors)), 0)
        pair = mx.concatenate([H_src, H_dst, E], axis=-1)
        return mx.maximum(self.edge_pair(pair), 0)

    def edge_auxiliary(
        self,
        owner: mx.array,
        strength: mx.array,
        outflow: mx.array,
        edge_pressure: mx.array,
        neighbors: mx.array,
        num_players: int,
    ) -> tuple[mx.array, mx.array]:
        H0, H = self._encode_nodes(
            owner, strength, outflow, edge_pressure, neighbors, num_players,
        )
        edge_h = self._edge_hidden(H0, H, outflow, edge_pressure, neighbors)
        return self.edge_type_head(edge_h), self.edge_channel_head(edge_h)

    def forward(
        self,
        owner: mx.array,
        strength: mx.array,
        outflow: mx.array,
        edge_pressure: mx.array,
        neighbors: mx.array,
        num_players: int,
    ) -> tuple[mx.array, mx.array]:
        H0, H = self._encode_nodes(
            owner, strength, outflow, edge_pressure, neighbors, num_players,
        )
        edge_h = self._edge_hidden(H0, H, outflow, edge_pressure, neighbors)
        scores = self.edge_score(edge_h)
        open_scores = scores[..., 0]
        clear_scores = scores[..., 1]

        G, S, N, _ = H.shape
        valid_slot = mx.broadcast_to((neighbors >= 0).reshape(1, 1, N, K), (G, S, N, K))
        source_mine = H0[..., 1].reshape(G, S, N, 1) > 0.5
        dst_is_dead = self._edge_features(H0, outflow, edge_pressure, neighbors)[..., 15] > 0.5
        set_mask = source_mine & valid_slot & mx.logical_not(dst_is_dead)
        clear_mask = source_mine & valid_slot
        open_logits = mx.where(set_mask, open_scores, -20.0)
        clear_logits = mx.where(clear_mask, clear_scores, -20.0)
        noop_logits = mx.where(
            source_mine.squeeze(-1),
            self.noop_head(H).squeeze(-1),
            0.0,
        )
        policy_logits = mx.concatenate(
            [open_logits, clear_logits, noop_logits[..., None]], axis=-1
        )

        v_per_cell = self.value_out(
            mx.maximum(self.value_hidden(H), 0)
        ).squeeze(-1)
        seat_idx = mx.arange(num_players).reshape(1, num_players, 1)
        is_mine = (owner.reshape(G, 1, N) == seat_idx).astype(mx.float32)
        denom = mx.maximum(is_mine.sum(axis=-1), 1.0)
        value = (v_per_cell * is_mine).sum(axis=-1) / denom
        return policy_logits, value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def make_actor_critic(name: str = "gnn") -> nn.Module:
    """Factory used by scripts so the node baseline remains the default."""
    if name == "gnn":
        return GNNActorCritic()
    if name == "edge":
        return EdgeAwareActorCritic()
    raise ValueError(f"unknown v2 model '{name}'")
