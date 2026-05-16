"""Pure NumPy derived edge features for flux v2.

This module is representation-only: it derives seat-relative directed-edge
facts from the pressure state and never changes reducer physics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import DEAD, K, MAX_EDGE, MAX_STRENGTH, NEUTRAL, State


EDGE_CATEGORY_NAMES: tuple[str, ...] = (
    "mine_to_enemy",
    "mine_to_neutral",
    "mine_to_friendly_relay",
    "mine_to_friendly_sink",
    "mine_to_friendly_fill",
    "enemy_to_mine",
    "enemy_to_enemy",
    "blocked",
)
EDGE_CATEGORY_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(EDGE_CATEGORY_NAMES)
}

EDGE_MINE_TO_ENEMY = EDGE_CATEGORY_INDEX["mine_to_enemy"]
EDGE_MINE_TO_NEUTRAL = EDGE_CATEGORY_INDEX["mine_to_neutral"]
EDGE_MINE_TO_FRIENDLY_RELAY = EDGE_CATEGORY_INDEX["mine_to_friendly_relay"]
EDGE_MINE_TO_FRIENDLY_SINK = EDGE_CATEGORY_INDEX["mine_to_friendly_sink"]
EDGE_MINE_TO_FRIENDLY_FILL = EDGE_CATEGORY_INDEX["mine_to_friendly_fill"]
EDGE_ENEMY_TO_MINE = EDGE_CATEGORY_INDEX["enemy_to_mine"]
EDGE_ENEMY_TO_ENEMY = EDGE_CATEGORY_INDEX["enemy_to_enemy"]
EDGE_BLOCKED = EDGE_CATEGORY_INDEX["blocked"]
NUM_EDGE_CATEGORIES = len(EDGE_CATEGORY_NAMES)

EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "source_strength",
    "source_headroom",
    "dest_strength",
    "dest_headroom",
    "outflow_active",
    "edge_pressure",
    "dest_active_outflow_count",
    "dest_is_mine",
    "dest_is_enemy",
    "dest_is_neutral",
    "dest_is_dead",
    "source_is_mine",
    "source_is_enemy",
    "source_is_neutral",
    "source_is_dead",
    "source_frontier",
    "dest_frontier",
)
EDGE_FEATURE_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(EDGE_FEATURE_NAMES)
}
NUM_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)

__all__ = [
    "EDGE_CATEGORY_NAMES",
    "EDGE_CATEGORY_INDEX",
    "EDGE_MINE_TO_ENEMY",
    "EDGE_MINE_TO_NEUTRAL",
    "EDGE_MINE_TO_FRIENDLY_RELAY",
    "EDGE_MINE_TO_FRIENDLY_SINK",
    "EDGE_MINE_TO_FRIENDLY_FILL",
    "EDGE_ENEMY_TO_MINE",
    "EDGE_ENEMY_TO_ENEMY",
    "EDGE_BLOCKED",
    "NUM_EDGE_CATEGORIES",
    "EDGE_FEATURE_NAMES",
    "EDGE_FEATURE_INDEX",
    "NUM_EDGE_FEATURES",
    "EdgeMasks",
    "EdgeFeatureBatch",
    "edge_category_one_hot",
    "build_edge_features_for_state",
    "build_edge_features",
]


@dataclass(frozen=True)
class EdgeMasks:
    """Seat-relative boolean masks, all shaped ``(G, S, N, K)``."""

    valid_destination: np.ndarray
    blocked: np.ndarray
    source_alive: np.ndarray
    source_owned: np.ndarray
    actionable: np.ndarray
    active_outflow: np.ndarray
    active_pressure: np.ndarray
    dest_mine: np.ndarray
    dest_enemy: np.ndarray
    dest_neutral: np.ndarray
    dest_dead: np.ndarray
    source_enemy: np.ndarray
    category_known: np.ndarray


@dataclass(frozen=True)
class EdgeFeatureBatch:
    """Derived edge features for batched v2 pressure states.

    ``features`` has shape ``(G, S, N, K, F)`` and ``category`` has shape
    ``(G, S, N, K)``. ``neighbors`` is the source slot table used to build the
    batch and keeps the destination ids attached to the facts.
    """

    features: np.ndarray
    category: np.ndarray
    category_one_hot: np.ndarray
    masks: EdgeMasks
    neighbors: np.ndarray


def edge_category_one_hot(category: np.ndarray) -> np.ndarray:
    """Return one-hot edge-category targets with final dim ``C``."""

    category = np.asarray(category, dtype=np.int32)
    if category.size and (
        int(category.min()) < 0 or int(category.max()) >= NUM_EDGE_CATEGORIES
    ):
        raise ValueError("edge category id out of range")
    return np.eye(NUM_EDGE_CATEGORIES, dtype=np.float32)[category]


def build_edge_features_for_state(state: State) -> EdgeFeatureBatch:
    """Build edge features for a single ``State`` as ``G=1``."""

    return build_edge_features(
        owner=state.owner[None, :],
        strength=state.strength[None, :],
        outflow=state.outflow[None, :, :],
        edge_pressure=state.edge_pressure[None, :, :],
        neighbors=state.neighbors,
        num_players=state.num_players,
    )


def build_edge_features(
    owner: np.ndarray,
    strength: np.ndarray,
    outflow: np.ndarray,
    edge_pressure: np.ndarray,
    neighbors: np.ndarray,
    num_players: int,
) -> EdgeFeatureBatch:
    """Derive seat-relative directed-edge facts.

    Args:
        owner: ``(G, N)`` int owners, with ``NEUTRAL``/``DEAD`` sentinels.
        strength: ``(G, N)`` float cell strengths.
        outflow: ``(G, N, K)`` bool persistent gates.
        edge_pressure: ``(G, N, K)`` float current directed pressure.
        neighbors: ``(N, K)`` destination ids, ``-1`` for off-grid.
        num_players: number of seats ``S``.
    """

    owner = np.asarray(owner, dtype=np.int32)
    strength = np.asarray(strength, dtype=np.float32)
    outflow = np.asarray(outflow, dtype=np.bool_)
    edge_pressure = np.asarray(edge_pressure, dtype=np.float32)
    neighbors = np.asarray(neighbors, dtype=np.int32)

    if owner.ndim != 2:
        raise ValueError(f"owner must be (G, N), got {owner.shape}")
    if strength.shape != owner.shape:
        raise ValueError(f"strength shape {strength.shape} != owner {owner.shape}")
    if neighbors.ndim != 2 or neighbors.shape[1] != K:
        raise ValueError(f"neighbors must be (N, {K}), got {neighbors.shape}")

    G, N = owner.shape
    if neighbors.shape[0] != N:
        raise ValueError(f"neighbors N {neighbors.shape[0]} != owner N {N}")
    if outflow.shape != (G, N, K):
        raise ValueError(f"outflow must be {(G, N, K)}, got {outflow.shape}")
    if edge_pressure.shape != (G, N, K):
        raise ValueError(
            f"edge_pressure must be {(G, N, K)}, got {edge_pressure.shape}"
        )
    if num_players <= 0:
        raise ValueError(f"num_players must be positive, got {num_players}")

    S = int(num_players)
    valid_nk = neighbors >= 0
    safe_neighbors = np.maximum(neighbors, 0)

    seat = np.arange(S, dtype=np.int32).reshape(1, S, 1, 1)
    valid = np.broadcast_to(valid_nk.reshape(1, 1, N, K), (G, S, N, K))

    src_owner = np.broadcast_to(owner.reshape(G, 1, N, 1), (G, S, N, K))
    src_strength = np.broadcast_to(
        strength.reshape(G, 1, N, 1), (G, S, N, K)
    )
    src_headroom = np.maximum(MAX_STRENGTH - src_strength, 0.0)
    active_outflow = np.broadcast_to(outflow.reshape(G, 1, N, K), (G, S, N, K))
    pressure = np.broadcast_to(edge_pressure.reshape(G, 1, N, K), (G, S, N, K))

    dst_owner_gnk = owner[:, safe_neighbors.reshape(-1)].reshape(G, N, K)
    dst_strength_gnk = strength[:, safe_neighbors.reshape(-1)].reshape(G, N, K)
    dst_active_count_gnk = (
        outflow.sum(axis=-1).astype(np.float32)[:, safe_neighbors.reshape(-1)]
        .reshape(G, N, K)
    )

    valid_gnk = valid_nk.reshape(1, N, K)
    dst_owner_gnk = np.where(valid_gnk, dst_owner_gnk, DEAD)
    dst_strength_gnk = np.where(valid_gnk, dst_strength_gnk, 0.0)
    dst_active_count_gnk = np.where(valid_gnk, dst_active_count_gnk, 0.0)

    dst_owner = np.broadcast_to(dst_owner_gnk.reshape(G, 1, N, K), (G, S, N, K))
    dst_strength = np.broadcast_to(
        dst_strength_gnk.reshape(G, 1, N, K), (G, S, N, K)
    )
    dst_headroom = np.maximum(MAX_STRENGTH - dst_strength, 0.0)
    dst_active_count = np.broadcast_to(
        dst_active_count_gnk.reshape(G, 1, N, K), (G, S, N, K)
    )

    source_dead = src_owner == DEAD
    source_neutral = src_owner == NEUTRAL
    source_alive = src_owner >= 0
    source_owned = source_alive & (src_owner == seat)
    source_enemy = source_alive & (src_owner != seat)

    dest_dead = valid & (dst_owner == DEAD)
    dest_neutral = valid & (dst_owner == NEUTRAL)
    dest_mine = valid & (dst_owner == seat)
    dest_enemy = valid & (dst_owner >= 0) & (dst_owner != seat)
    blocked = (~valid) | source_dead | dest_dead
    category_known = source_alive & (~blocked)
    actionable = source_owned & (~blocked)
    active_pressure = active_outflow & (pressure > 0.0)

    source_frontier_node, dest_frontier = _frontier_masks(
        owner=owner,
        neighbors=neighbors,
        safe_neighbors=safe_neighbors,
        valid_nk=valid_nk,
        seat=seat,
    )
    source_frontier = np.broadcast_to(
        source_frontier_node.reshape(G, S, N, 1), (G, S, N, K)
    )

    category = np.full((G, S, N, K), EDGE_BLOCKED, dtype=np.int32)

    mine_live = source_owned & (~blocked)
    category[mine_live & dest_enemy] = EDGE_MINE_TO_ENEMY
    category[mine_live & dest_neutral] = EDGE_MINE_TO_NEUTRAL

    friendly = mine_live & dest_mine
    relay = friendly & (dst_active_count > 0.0)
    sink = friendly & (~relay) & (dst_strength >= MAX_STRENGTH)
    fill = friendly & (~relay) & (dst_strength < MAX_STRENGTH)
    category[relay] = EDGE_MINE_TO_FRIENDLY_RELAY
    category[sink] = EDGE_MINE_TO_FRIENDLY_SINK
    category[fill] = EDGE_MINE_TO_FRIENDLY_FILL

    enemy_live = source_enemy & (~blocked)
    category[enemy_live & dest_mine] = EDGE_ENEMY_TO_MINE
    category[enemy_live & (~dest_mine)] = EDGE_ENEMY_TO_ENEMY

    features = np.stack(
        [
            src_strength / float(MAX_STRENGTH),
            src_headroom / float(MAX_STRENGTH),
            dst_strength / float(MAX_STRENGTH),
            dst_headroom / float(MAX_STRENGTH),
            active_outflow.astype(np.float32),
            pressure / float(MAX_EDGE),
            dst_active_count / float(K),
            dest_mine.astype(np.float32),
            dest_enemy.astype(np.float32),
            dest_neutral.astype(np.float32),
            dest_dead.astype(np.float32),
            source_owned.astype(np.float32),
            source_enemy.astype(np.float32),
            source_neutral.astype(np.float32),
            source_dead.astype(np.float32),
            source_frontier.astype(np.float32),
            dest_frontier.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    masks = EdgeMasks(
        valid_destination=valid.copy(),
        blocked=blocked.copy(),
        source_alive=source_alive.copy(),
        source_owned=source_owned.copy(),
        actionable=actionable.copy(),
        active_outflow=active_outflow.copy(),
        active_pressure=active_pressure.copy(),
        dest_mine=dest_mine.copy(),
        dest_enemy=dest_enemy.copy(),
        dest_neutral=dest_neutral.copy(),
        dest_dead=dest_dead.copy(),
        source_enemy=source_enemy.copy(),
        category_known=category_known.copy(),
    )
    return EdgeFeatureBatch(
        features=features,
        category=category,
        category_one_hot=edge_category_one_hot(category),
        masks=masks,
        neighbors=neighbors.copy(),
    )


def _frontier_masks(
    *,
    owner: np.ndarray,
    neighbors: np.ndarray,
    safe_neighbors: np.ndarray,
    valid_nk: np.ndarray,
    seat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(source_node_frontier, dest_edge_frontier)`` masks."""

    G, N = owner.shape
    S = seat.shape[1]
    neighbor_owner = owner[:, safe_neighbors.reshape(-1)].reshape(G, N, K)
    valid = valid_nk.reshape(1, N, K)
    neighbor_owner = np.where(valid, neighbor_owner, DEAD)

    node_owner = owner.reshape(G, 1, N, 1)
    seat_b = seat
    node_live = node_owner >= 0
    node_mine = node_live & (node_owner == seat_b)
    neighbor_live = neighbor_owner.reshape(G, 1, N, K) != DEAD
    neighbor_mine = neighbor_owner.reshape(G, 1, N, K) == seat_b

    mine_next_to_nonfriendly = node_mine & neighbor_live & (~neighbor_mine)
    nonmine_next_to_mine = (~node_mine) & node_live & neighbor_mine
    node_frontier = (
        mine_next_to_nonfriendly | nonmine_next_to_mine
    ).any(axis=-1)

    dst_frontier_gsnk = np.zeros((G, S, N, K), dtype=np.bool_)
    for k in range(K):
        d = safe_neighbors[:, k]
        dst_frontier_gsnk[:, :, :, k] = node_frontier[:, :, d]
        dst_frontier_gsnk[:, :, :, k] &= valid_nk.reshape(1, 1, N, K)[:, :, :, k]
    return node_frontier, dst_frontier_gsnk
