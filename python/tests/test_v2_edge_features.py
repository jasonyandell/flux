from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import (
    DEAD,
    K,
    MAX_EDGE,
    MAX_STRENGTH,
    NEUTRAL,
    OPPOSITE_SLOT,
    State,
    make_board,
)
from flux_v2.edge_features import (
    EDGE_BLOCKED,
    EDGE_CATEGORY_NAMES,
    EDGE_ENEMY_TO_ENEMY,
    EDGE_ENEMY_TO_MINE,
    EDGE_FEATURE_INDEX,
    EDGE_MINE_TO_ENEMY,
    EDGE_MINE_TO_FRIENDLY_FILL,
    EDGE_MINE_TO_FRIENDLY_RELAY,
    EDGE_MINE_TO_FRIENDLY_SINK,
    EDGE_MINE_TO_NEUTRAL,
    NUM_EDGE_CATEGORIES,
    NUM_EDGE_FEATURES,
    build_edge_features_for_state,
)


def _center_board() -> tuple[State, int, np.ndarray]:
    s = make_board(radius=1, num_players=2)
    s.owner[:] = NEUTRAL
    s.strength[:] = 10.0
    s.outflow[:] = False
    s.edge_pressure[:] = 0.0
    center = int(np.where((s.coord[:, 0] == 0) & (s.coord[:, 1] == 0))[0][0])
    nbrs = s.neighbors[center].copy()
    assert (nbrs >= 0).all()
    return s, center, nbrs


def _slot_to(s: State, src: int, dst: int) -> int:
    for k in range(K):
        if int(s.neighbors[src, k]) == dst:
            return k
    raise AssertionError(f"{src} has no edge to {dst}")


def _first_live_slot_not_to(s: State, src: int, avoid: int) -> tuple[int, int]:
    for k in range(K):
        d = int(s.neighbors[src, k])
        if d >= 0 and d != avoid:
            return k, d
    raise AssertionError(f"{src} has no alternate live edge")


def _first_offgrid_slot(s: State, src: int) -> int:
    for k in range(K):
        if int(s.neighbors[src, k]) < 0:
            return k
    raise AssertionError(f"{src} has no off-grid slot")


def _cat(s: State, seat: int, src: int, slot: int) -> int:
    batch = build_edge_features_for_state(s)
    return int(batch.category[0, seat, src, slot])


def test_edge_feature_shapes_and_one_hot_targets():
    s, center, nbrs = _center_board()
    s.owner[center] = 0
    s.owner[nbrs[0]] = 1

    batch = build_edge_features_for_state(s)

    assert batch.features.shape == (1, 2, s.N, K, NUM_EDGE_FEATURES)
    assert batch.category.shape == (1, 2, s.N, K)
    assert batch.category_one_hot.shape == (1, 2, s.N, K, NUM_EDGE_CATEGORIES)
    assert batch.category_one_hot.dtype == np.float32
    assert batch.masks.source_owned.shape == (1, 2, s.N, K)
    assert batch.neighbors.shape == (s.N, K)
    assert EDGE_CATEGORY_NAMES[batch.category[0, 0, center, 0]] == "mine_to_enemy"


def test_mine_edge_categories_and_continuous_channels():
    s, center, nbrs = _center_board()
    s.owner[center] = 0
    s.strength[center] = 0.7 * MAX_STRENGTH

    enemy, neutral, relay, sink, fill, dead = map(int, nbrs)
    s.owner[enemy] = 1
    s.strength[enemy] = 0.4 * MAX_STRENGTH
    s.owner[neutral] = NEUTRAL
    s.strength[neutral] = 15.0
    s.owner[relay] = 0
    s.strength[relay] = MAX_STRENGTH
    relay_slot, _ = _first_live_slot_not_to(s, relay, center)
    s.outflow[relay, relay_slot] = True
    s.owner[sink] = 0
    s.strength[sink] = MAX_STRENGTH
    s.owner[fill] = 0
    s.strength[fill] = 20.0
    s.owner[dead] = DEAD
    s.strength[dead] = 0.0

    s.outflow[center, 0] = True
    s.edge_pressure[center, 0] = 25.0
    s.edge_pressure[center, 1] = 15.0

    batch = build_edge_features_for_state(s)
    category = batch.category[0, 0, center]
    assert int(category[0]) == EDGE_MINE_TO_ENEMY
    assert int(category[1]) == EDGE_MINE_TO_NEUTRAL
    assert int(category[2]) == EDGE_MINE_TO_FRIENDLY_RELAY
    assert int(category[3]) == EDGE_MINE_TO_FRIENDLY_SINK
    assert int(category[4]) == EDGE_MINE_TO_FRIENDLY_FILL
    assert int(category[5]) == EDGE_BLOCKED

    f = batch.features[0, 0, center]
    assert np.isclose(f[0, EDGE_FEATURE_INDEX["source_strength"]], 0.70)
    assert np.isclose(f[0, EDGE_FEATURE_INDEX["source_headroom"]], 0.30)
    assert np.isclose(f[0, EDGE_FEATURE_INDEX["dest_strength"]], 0.40)
    assert np.isclose(f[0, EDGE_FEATURE_INDEX["dest_headroom"]], 0.60)
    assert f[0, EDGE_FEATURE_INDEX["outflow_active"]] == 1.0
    assert np.isclose(f[0, EDGE_FEATURE_INDEX["edge_pressure"]], 25.0 / MAX_EDGE)
    assert f[1, EDGE_FEATURE_INDEX["outflow_active"]] == 0.0
    assert np.isclose(f[1, EDGE_FEATURE_INDEX["edge_pressure"]], 15.0 / MAX_EDGE)
    assert np.isclose(f[2, EDGE_FEATURE_INDEX["dest_active_outflow_count"]], 1.0 / K)
    assert f[0, EDGE_FEATURE_INDEX["dest_is_enemy"]] == 1.0
    assert f[1, EDGE_FEATURE_INDEX["dest_is_neutral"]] == 1.0
    assert f[5, EDGE_FEATURE_INDEX["dest_is_dead"]] == 1.0
    assert f[0, EDGE_FEATURE_INDEX["source_is_mine"]] == 1.0
    assert batch.masks.actionable[0, 0, center, 0]
    assert not batch.masks.actionable[0, 0, center, 5]


def test_enemy_categories_are_seat_relative():
    s, center, nbrs = _center_board()
    enemy_src = int(nbrs[0])
    s.owner[center] = 0
    s.owner[enemy_src] = 1

    k_to_mine = _slot_to(s, enemy_src, center)
    k_to_other, other = _first_live_slot_not_to(s, enemy_src, center)
    s.owner[other] = 1

    batch = build_edge_features_for_state(s)
    assert int(batch.category[0, 0, enemy_src, k_to_mine]) == EDGE_ENEMY_TO_MINE
    assert int(batch.category[0, 0, enemy_src, k_to_other]) == EDGE_ENEMY_TO_ENEMY
    assert (
        batch.features[
            0, 0, enemy_src, k_to_mine, EDGE_FEATURE_INDEX["source_is_enemy"]
        ]
        == 1.0
    )
    assert (
        batch.features[
            0, 0, enemy_src, k_to_mine, EDGE_FEATURE_INDEX["dest_is_mine"]
        ]
        == 1.0
    )


def test_blocked_dead_source_and_off_grid_slots():
    s, center, nbrs = _center_board()
    s.owner[center] = 0
    dead_dst = int(nbrs[0])
    s.owner[dead_dst] = DEAD
    assert _cat(s, 0, center, 0) == EDGE_BLOCKED

    perim = dead_dst
    s.owner[perim] = 0
    off_slot = _first_offgrid_slot(s, perim)
    batch = build_edge_features_for_state(s)
    assert int(batch.category[0, 0, perim, off_slot]) == EDGE_BLOCKED
    assert not batch.masks.valid_destination[0, 0, perim, off_slot]
    assert batch.masks.blocked[0, 0, perim, off_slot]

    live_slot, _ = _first_live_slot_not_to(s, perim, -1)
    s.owner[perim] = DEAD
    batch = build_edge_features_for_state(s)
    assert int(batch.category[0, 0, perim, live_slot]) == EDGE_BLOCKED
    assert not batch.masks.source_alive[0, 0, perim, live_slot]


def test_owner_flip_changes_category_without_touching_outflow():
    s, center, nbrs = _center_board()
    target = int(nbrs[0])
    s.owner[center] = 0
    s.strength[center] = 80.0
    s.owner[target] = 0
    s.strength[target] = 30.0
    s.outflow[center, 0] = True

    assert _cat(s, 0, center, 0) == EDGE_MINE_TO_FRIENDLY_FILL

    s.owner[target] = 1
    assert _cat(s, 0, center, 0) == EDGE_MINE_TO_ENEMY
    assert s.outflow[center, 0]

    s.owner[center] = 1
    s.owner[target] = 0
    assert _cat(s, 0, center, 0) == EDGE_ENEMY_TO_MINE
    assert s.outflow[center, 0]


def test_edge_pressure_agrees_with_existing_nonfriendly_inbound_definition():
    s, center, nbrs = _center_board()
    s.owner[center] = 0
    s.owner[nbrs[0]] = 1
    s.owner[nbrs[1]] = 0
    s.owner[nbrs[2]] = 1
    s.owner[nbrs[3]] = NEUTRAL
    s.owner[nbrs[4]] = 0
    s.owner[nbrs[5]] = DEAD

    src_enemy_a = int(nbrs[0])
    src_friend = int(nbrs[1])
    src_enemy_b = int(nbrs[2])
    k_enemy_a = _slot_to(s, src_enemy_a, center)
    k_friend = _slot_to(s, src_friend, center)
    k_enemy_b = _slot_to(s, src_enemy_b, center)
    s.edge_pressure[src_enemy_a, k_enemy_a] = 11.0
    s.edge_pressure[src_friend, k_friend] = 7.0
    s.edge_pressure[src_enemy_b, k_enemy_b] = 5.0

    batch = build_edge_features_for_state(s)
    from_edges = np.zeros((1, 2, s.N), dtype=np.float32)
    pressure_channel = EDGE_FEATURE_INDEX["edge_pressure"]
    for src in range(s.N):
        for k in range(K):
            dst = int(s.neighbors[src, k])
            if dst < 0:
                continue
            for seat in range(2):
                if batch.masks.blocked[0, seat, src, k]:
                    continue
                if batch.masks.source_owned[0, seat, src, k]:
                    continue
                from_edges[0, seat, dst] += (
                    batch.features[0, seat, src, k, pressure_channel] * MAX_EDGE
                )

    reference = np.zeros((1, 2, s.N), dtype=np.float32)
    for dst in range(s.N):
        for k in range(K):
            src = int(s.neighbors[dst, k])
            if src < 0 or int(s.owner[src]) == DEAD:
                continue
            opp = int(OPPOSITE_SLOT[k])
            for seat in range(2):
                if int(s.owner[src]) != seat:
                    reference[0, seat, dst] += s.edge_pressure[src, opp]

    np.testing.assert_allclose(from_edges, reference)
    assert from_edges[0, 0, center] == 16.0
    assert from_edges[0, 1, center] == 7.0
