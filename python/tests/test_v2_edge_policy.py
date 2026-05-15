"""Tests for v2 edge-aware policy helpers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from flux_v2 import K, MAX_STRENGTH, NEUTRAL, OPPOSITE_SLOT, make_board
from flux_v2.ppo import (
    EDGE_CHANNEL_NAMES,
    EDGE_TYPE_NAMES,
    EdgeAwareActorCritic,
    build_edge_auxiliary_targets_np,
    make_actor_critic,
)


def _valid_slots(state, cell: int) -> list[int]:
    return [k for k in range(K) if int(state.neighbors[cell, k]) >= 0]


def test_edge_auxiliary_targets_cover_core_categories():
    state = make_board(radius=3, num_players=3)
    src = max(range(state.N), key=lambda c: len(_valid_slots(state, c)))
    state.owner[src] = 0
    state.strength[src] = 40.0
    slots = _valid_slots(state, src)
    assert len(slots) >= 5

    enemy_slot, neutral_slot, relay_slot, sink_slot, fill_slot = slots[:5]
    enemy_dst = int(state.neighbors[src, enemy_slot])
    neutral_dst = int(state.neighbors[src, neutral_slot])
    relay_dst = int(state.neighbors[src, relay_slot])
    sink_dst = int(state.neighbors[src, sink_slot])
    fill_dst = int(state.neighbors[src, fill_slot])

    state.owner[enemy_dst] = 1
    state.owner[neutral_dst] = NEUTRAL
    state.owner[relay_dst] = 0
    state.outflow[relay_dst, _valid_slots(state, relay_dst)[0]] = True
    state.owner[sink_dst] = 0
    state.strength[sink_dst] = MAX_STRENGTH
    state.outflow[sink_dst] = False
    state.owner[fill_dst] = 0
    state.strength[fill_dst] = 25.0
    state.outflow[fill_dst] = False

    back_src = enemy_dst
    back_slot = int(OPPOSITE_SLOT[enemy_slot])
    state.owner[src] = 0
    state.owner[back_src] = 1

    labels, channels, mask = build_edge_auxiliary_targets_np(
        state.owner.reshape(1, -1),
        state.strength.reshape(1, -1),
        state.outflow.reshape(1, state.N, K),
        state.edge_pressure.reshape(1, state.N, K),
        state.neighbors,
        state.num_players,
    )
    seat_labels = labels[0, 0]
    assert EDGE_TYPE_NAMES[seat_labels[src, enemy_slot]] == "mine_to_enemy"
    assert EDGE_TYPE_NAMES[seat_labels[src, neutral_slot]] == "mine_to_neutral"
    assert EDGE_TYPE_NAMES[seat_labels[src, relay_slot]] == "mine_to_friendly_relay"
    assert EDGE_TYPE_NAMES[seat_labels[src, sink_slot]] == "mine_to_friendly_sink"
    assert EDGE_TYPE_NAMES[seat_labels[src, fill_slot]] == "mine_to_friendly_fill"
    assert EDGE_TYPE_NAMES[seat_labels[back_src, back_slot]] == "enemy_to_mine"
    assert mask[0, 0, src, enemy_slot]

    attack_idx = EDGE_CHANNEL_NAMES.index("attack_flow")
    threat_idx = EDGE_CHANNEL_NAMES.index("threat_in")
    assert channels[0, 0, src, enemy_slot, attack_idx] == 1.0
    assert channels[0, 0, back_src, back_slot, threat_idx] == 1.0


def test_edge_model_forward_and_aux_shapes():
    state = make_board(radius=3, num_players=3)
    owner = mx.array(state.owner.reshape(1, -1))
    strength = mx.array(state.strength.reshape(1, -1))
    outflow = mx.array(state.outflow.reshape(1, state.N, K))
    edge_pressure = mx.array(state.edge_pressure.reshape(1, state.N, K))
    neighbors = mx.array(state.neighbors)

    model = EdgeAwareActorCritic()
    logits, value = model(owner, strength, outflow, edge_pressure, neighbors, state.num_players)
    type_logits, channel_pred = model.edge_auxiliary(
        owner, strength, outflow, edge_pressure, neighbors, state.num_players,
    )
    mx.eval(logits, value, type_logits, channel_pred)

    assert logits.shape == (1, state.num_players, state.N, 13)
    assert value.shape == (1, state.num_players)
    assert type_logits.shape[:4] == (1, state.num_players, state.N, K)
    assert channel_pred.shape[:4] == (1, state.num_players, state.N, K)
    assert np.isfinite(np.array(logits)).all()
    assert np.isfinite(np.array(value)).all()


def test_model_factory_keeps_gnn_default_and_exposes_edge():
    assert make_actor_critic("gnn").__class__.__name__ == "GNNActorCritic"
    assert make_actor_critic("edge").__class__.__name__ == "EdgeAwareActorCritic"
