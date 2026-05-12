"""Smoke test for MLX step: runs a few ticks against the NumPy reference and
prints owner/strength deltas. Not a parity test (float32 vs float64), just a
sanity check that step_mlx executes and produces vaguely-correct state.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from flux import (
    Action,
    apply_action,
    ai_think,
    build_neighbor_table,
    make_initial_state,
    mulberry32,
    random_genome,
    step,
)
from flux.mlx_step import step_mlx


def state_to_mlx(state):
    owners_np = np.array(
        [-1 if n.owner is None else n.owner for n in state.nodes], dtype=np.int32
    )
    strengths_np = np.array([n.strength for n in state.nodes], dtype=np.float32)
    flow_src = np.array([f.src for f in state.flows], dtype=np.int32)
    flow_dst = np.array([f.dst for f in state.flows], dtype=np.int32)
    flow_player = np.array([f.player for f in state.flows], dtype=np.int32)
    return (
        mx.array(owners_np),
        mx.array(strengths_np),
        mx.array(flow_src),
        mx.array(flow_dst),
        mx.array(flow_player),
    )


def main() -> None:
    state = make_initial_state(radius=9, distance=2, num_players=4)
    rng = mulberry32(42)
    genome = random_genome(rng, std=2.0)
    nb = build_neighbor_table(state)

    # Seed a few flows.
    for a in [
        Action(kind="toggleFlow", src=10, dst=11, player=0),
        Action(kind="toggleFlow", src=10, dst=21, player=0),
    ]:
        state = apply_action(state, a)

    t0 = time.time()
    np_state = state
    mlx_state = state
    for t in range(1, 101):
        # Both branches run the same actions, then step independently.
        for p in range(state.num_players):
            actions = ai_think(np_state, p, genome, nb)
            for a in actions:
                np_state = apply_action(np_state, a)
                mlx_state = apply_action(mlx_state, a)
        # NumPy step
        np_state = step(np_state, 0.1)
        # MLX step
        owner, strength, src, dst, player = state_to_mlx(mlx_state)
        new_owner, new_strength, alive_mask = step_mlx(
            owner, strength, src, dst, player, mlx_state.num_players, 0.1
        )
        mx.eval(new_owner, new_strength, alive_mask)
        # Rebuild python state for the next iteration.
        new_owner_np = np.array(new_owner)
        new_strength_np = np.array(new_strength)
        alive_np = np.array(alive_mask)
        # Filter flows.
        new_flows = tuple(
            f for f, keep in zip(mlx_state.flows, alive_np.tolist()) if keep
        )
        from flux.state import GameNode, GameState  # local import
        new_nodes = tuple(
            GameNode(
                id=n.id,
                pos=n.pos,
                owner=None if int(new_owner_np[i]) == -1 else int(new_owner_np[i]),
                strength=float(new_strength_np[i]),
            )
            for i, n in enumerate(mlx_state.nodes)
        )
        mlx_state = GameState(
            nodes=new_nodes,
            edges=mlx_state.edges,
            flows=new_flows,
            tick=mlx_state.tick + 1,
            num_players=mlx_state.num_players,
        )

    dt = time.time() - t0
    # Compare endpoints.
    max_strength_diff = 0.0
    owner_mismatches = 0
    for i in range(len(np_state.nodes)):
        a = np_state.nodes[i]
        b = mlx_state.nodes[i]
        ao = -1 if a.owner is None else a.owner
        bo = -1 if b.owner is None else b.owner
        if ao != bo:
            owner_mismatches += 1
        max_strength_diff = max(max_strength_diff, abs(a.strength - b.strength))
    print(
        f"20 ticks: maxΔstrength={max_strength_diff:.4f}  "
        f"ownerMismatches={owner_mismatches}/{len(np_state.nodes)}  "
        f"flows np={len(np_state.flows)} mlx={len(mlx_state.flows)}  "
        f"elapsed={dt:.3f}s"
    )


if __name__ == "__main__":
    main()
