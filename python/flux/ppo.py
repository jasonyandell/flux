"""PPO + GNN actor-critic for flux.

Architecture: same 2-layer GCN backbone as v3, with two heads bolted on:
    - Policy head: per-cell linear → 19 action logits.
    - Value head: per-cell scalar → mean over seat-owned cells → V(state, seat).

All seats share parameters (one policy applied per-seat). The 4-channel
per-(seat, cell) input is seat-relative, so the same parameters work for any
seat — no per-seat copies needed.

This file is the network + loss only. Rollout collection and the training
loop live in `scripts/train_ppo.py`.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .state import MAX_STRENGTH

NEIGHBOR_STRIDE = 18              # matches mlx_genome.NEIGHBOR_STRIDE
IN_DIM = 5                       # strength, is_mine, is_enemy, is_neutral, is_dead
HIDDEN = 32
POLICY_OUT = 19
VALUE_HIDDEN = 16


def _aggregate_neighbors(H: mx.array, graph_neighbors: mx.array) -> mx.array:
    """Mean-aggregate H over each cell's graph neighbors. Empty slots (-1) do
    not contribute and don't count toward the divisor.
    H: (..., N, F) — leading axes preserved.
    graph_neighbors: (N * NEIGHBOR_STRIDE,) int32, -1 = empty.
    Returns: same shape as H."""
    *lead, N, F = H.shape
    STRIDE = NEIGHBOR_STRIDE
    nb_ids = graph_neighbors.reshape(N, STRIDE)
    nb_valid = nb_ids >= 0
    safe = mx.maximum(nb_ids, 0).reshape(-1)             # (N*STRIDE,)
    gathered = mx.take(H, safe, axis=-2)                 # (*lead, N*STRIDE, F)
    gathered = gathered.reshape(*lead, N, STRIDE, F)
    mask_shape = [1] * len(lead) + [N, STRIDE, 1]
    mask = nb_valid.astype(mx.float32).reshape(mask_shape)
    summed = (gathered * mask).sum(axis=-2)               # (*lead, N, F)
    count_shape = [1] * len(lead) + [N, 1]
    count = mx.maximum(
        nb_valid.astype(mx.float32).sum(axis=-1).reshape(count_shape),
        1.0,
    )
    return summed / count


def build_features(
    owner: mx.array,        # (G, N) int32
    strength: mx.array,     # (G, N) float32
    num_players: int,
    dead_mask: mx.array | None = None,  # (G, N) bool, optional
) -> mx.array:
    """Returns (G, S, N, 5) per-(seat, cell) features.

    Channels: strength_norm, is_mine, is_enemy, is_neutral, is_dead.
    Dead cells are presented as is_neutral=0, is_dead=1 so the policy can
    distinguish "untouchable obstacle" from "capturable empty cell."
    """
    G, N = owner.shape
    S = num_players
    out_shape = (G, S, N)
    seat_idx = mx.arange(S).reshape(1, S, 1)
    owner_b = owner.reshape(G, 1, N)
    if dead_mask is None:
        is_dead_2d = mx.zeros((G, N), dtype=mx.bool_)
    else:
        is_dead_2d = dead_mask
    is_dead_b = is_dead_2d.reshape(G, 1, N)
    not_dead_b = mx.logical_not(is_dead_b)
    is_mine = mx.broadcast_to(
        ((owner_b == seat_idx) & not_dead_b).astype(mx.float32), out_shape
    )
    is_neutral = mx.broadcast_to(
        ((owner_b == -1) & not_dead_b).astype(mx.float32), out_shape
    )
    is_enemy = mx.broadcast_to(
        ((owner_b != seat_idx) & (owner_b != -1) & not_dead_b).astype(mx.float32), out_shape
    )
    is_dead = mx.broadcast_to(is_dead_b.astype(mx.float32), out_shape)
    strength_norm = mx.broadcast_to(
        (strength / MAX_STRENGTH).reshape(G, 1, N).astype(mx.float32), out_shape
    )
    return mx.stack([strength_norm, is_mine, is_enemy, is_neutral, is_dead], axis=-1)


class GNNActorCritic(nn.Module):
    """2-layer GCN backbone + policy and value heads. Shared across seats."""

    def __init__(self) -> None:
        super().__init__()
        # MP layer 1
        self.w1_self = nn.Linear(IN_DIM, HIDDEN, bias=False)
        self.w1_neigh = nn.Linear(IN_DIM, HIDDEN, bias=True)
        # MP layer 2
        self.w2_self = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w2_neigh = nn.Linear(HIDDEN, HIDDEN, bias=True)
        # Policy head (per-cell)
        self.policy_head = nn.Linear(HIDDEN, POLICY_OUT)
        # Value head (per-cell, pooled to per-seat after)
        self.value_hidden = nn.Linear(HIDDEN, VALUE_HIDDEN)
        self.value_out = nn.Linear(VALUE_HIDDEN, 1)

    def forward(
        self,
        owner: mx.array,           # (G, N) int32
        strength: mx.array,        # (G, N) float32
        graph_neighbors: mx.array, # (N * STRIDE,) int32
        num_players: int,
        dead_mask: mx.array | None = None,  # (G, N) bool
    ) -> tuple[mx.array, mx.array]:
        """Returns (policy_logits, value):
            policy_logits: (G, S, N, 19)
            value: (G, S) — V(state, seat), mean-pooled over seat-owned cells.
        """
        H0 = build_features(owner, strength, num_players, dead_mask)            # (G, S, N, 5)
        H0_agg = _aggregate_neighbors(H0, graph_neighbors)
        H1 = mx.maximum(self.w1_self(H0) + self.w1_neigh(H0_agg), 0)           # (G, S, N, H)
        H1_agg = _aggregate_neighbors(H1, graph_neighbors)
        H2 = mx.maximum(self.w2_self(H1) + self.w2_neigh(H1_agg), 0)
        policy_logits = self.policy_head(H2)                                   # (G, S, N, 19)
        v_per_cell = self.value_out(
            mx.maximum(self.value_hidden(H2), 0)
        ).squeeze(-1)                                                          # (G, S, N)
        # Pool value over each seat's owned cells (mean).
        G, N = owner.shape
        S = num_players
        seat_idx = mx.arange(S).reshape(1, S, 1)
        is_mine = (owner.reshape(G, 1, N) == seat_idx).astype(mx.float32)
        denom = mx.maximum(is_mine.sum(axis=-1), 1.0)                          # (G, S)
        value = (v_per_cell * is_mine).sum(axis=-1) / denom                    # (G, S)
        return policy_logits, value

    # Allow `model(...)` calls.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
