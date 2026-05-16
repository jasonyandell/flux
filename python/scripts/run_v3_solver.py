"""Run v2 algorithmic solvers on a topologically-spherical board.

Like `run_v2_solver.py` but the board geometry is a subdivided icosahedron
(geodesic sphere) instead of a flat hex disc:

    python scripts/run_v2_sphere_solver.py --subdiv 4 --num-players 6 \
        --games 1 --max-ticks 10000 --dead-frac 0.4 --write-replay

Sphere geometry (3D vertex positions and per-cell adjacency table) ships
to the viewer in the FLXR replay's metadata, so the displayer can
reconstruct the exact same graph the sim played on.
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v3 import (
    ACTION_NOOP,
    DEAD,
    K,
    NEUTRAL,
    apply_actions,
    copy_state,
    tick,
)
from flux_v3.graph import (
    carve_seat_connectors,
    seats_mutually_reachable,
)
from flux_v3.replay import ReplayHeader, ReplayWriter, append_index, state_to_frame
from flux_v3.solver import solver_actions
from flux_v3.solver_lightning import lightning_solver_actions
from flux_v3.sphere_graph import make_sphere_board
from flux_v3.state import MAX_STRENGTH


# Reuse the lightning solver mode wrappers. The function names match the
# entries in the hex runner's SOLVERS dict so users can pass familiar names.
def _lightning_sum(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="sum")


def _lightning_sum_long(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="sum", gamma=0.94)


def _lightning_attn(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="attn")


def _lightning_vortex(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="vortex")


def _lightning_flood(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="flood")


def _lightning_pulse_stagger(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="pulse_stagger")


def _lightning_chase(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="chase")


def _lightning_wave_keep_attack_long(state, seat, rng=None):
    return lightning_solver_actions(
        state, seat, rng=rng, mode="wave_keep_attack", gamma=0.94,
    )


SOLVERS = {
    "bfs": solver_actions,
    "lightning": lightning_solver_actions,
    "lightning_sum": _lightning_sum,
    "lightning_sum_long": _lightning_sum_long,
    "lightning_attn": _lightning_attn,
    "lightning_vortex": _lightning_vortex,
    "lightning_flood": _lightning_flood,
    "lightning_pulse_stagger": _lightning_pulse_stagger,
    "lightning_chase": _lightning_chase,
    "lightning_wave_keep_attack_long": _lightning_wave_keep_attack_long,
}


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "public" / "v3" / "replays"


def _sample_dead_connected(
    V: int, num_dead: int, neighbors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedy uniform-random dead-cell sampling that preserves live-graph
    connectivity. Mirrors graph.random_seat_and_dead's connectivity guard
    but doesn't pick seats — sphere boards pick seats greedy farthest-point."""
    if num_dead <= 0:
        return np.zeros(0, dtype=np.int32)
    is_dead = np.zeros(V, dtype=np.bool_)
    placed = 0
    candidates = np.arange(V)
    rng.shuffle(candidates)

    def _live_connected() -> bool:
        # Pick any live start; BFS through live cells.
        start = int(np.argmax(~is_dead))
        if is_dead[start]:
            return True
        visited = np.zeros(V, dtype=np.bool_)
        visited[start] = True
        stack = [start]
        while stack:
            c = stack.pop()
            for k in range(K):
                d = int(neighbors[c, k])
                if d >= 0 and not is_dead[d] and not visited[d]:
                    visited[d] = True
                    stack.append(d)
        return int(visited.sum()) == int((~is_dead).sum())

    for c in candidates:
        if placed >= num_dead:
            break
        c = int(c)
        if is_dead[c]:
            continue
        is_dead[c] = True
        if _live_connected():
            placed += 1
        else:
            is_dead[c] = False
    return np.where(is_dead)[0].astype(np.int32)


def _greedy_farthest_seats(
    V: int, num_players: int, pos3d: np.ndarray,
    is_dead: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Greedy farthest-point on great-circle distance, restricted to live
    cells. Spreads seats maximally without solving a full n-body problem."""
    live_pool = np.where(~is_dead)[0]
    if len(live_pool) < num_players:
        raise RuntimeError(f"only {len(live_pool)} live cells for {num_players} seats")
    first = int(rng.choice(live_pool))
    chosen = [first]
    in_set = {first}
    pos_norm = pos3d / np.linalg.norm(pos3d, axis=1, keepdims=True)
    for _ in range(num_players - 1):
        # Min angular distance from each candidate to any chosen seat.
        cand_pos = pos_norm[live_pool]
        chosen_pos = pos_norm[np.asarray(chosen)]
        cos = np.clip(cand_pos @ chosen_pos.T, -1.0, 1.0)         # (M, len(chosen))
        ang = np.arccos(cos)
        min_d = ang.min(axis=1)
        # Mask out already-chosen.
        for i, ci in enumerate(live_pool):
            if int(ci) in in_set:
                min_d[i] = -1.0
        pick = live_pool[int(np.argmax(min_d))]
        chosen.append(int(pick))
        in_set.add(int(pick))
    return np.asarray(chosen, dtype=np.int32)


def _build_initial_state(
    subdiv: int, num_players: int, dead_frac: float,
    rng: np.random.Generator,
):
    """Build a sphere board with `dead_frac` dead cells, then carve the live
    subgraph to one connected component. Seats placed by greedy farthest-point.
    """
    base, pos3d = make_sphere_board(subdiv=subdiv, num_players=num_players)
    V = base.N
    num_dead = int(round(dead_frac * V))

    # Sample dead with connectivity preservation (best effort).
    dead = _sample_dead_connected(V, num_dead, base.neighbors, rng)
    is_dead = np.zeros(V, dtype=np.bool_)
    if len(dead) > 0:
        is_dead[dead] = True

    seats = _greedy_farthest_seats(V, num_players, pos3d, is_dead, rng)

    # Belt-and-suspenders: even with the per-cell connectivity guard, the
    # carve pass guarantees seat-mutual-reachability after a possible bad
    # accident. Typically a no-op here.
    if not seats_mutually_reachable(seats, dead, base.neighbors):
        new_dead, carved = carve_seat_connectors(seats, dead, base.neighbors)
        if len(carved) > 0:
            print(f"  (carve: bridged {len(carved)} cells to connect seats)")
        dead = new_dead
        is_dead = np.zeros(V, dtype=np.bool_)
        if len(dead) > 0:
            is_dead[dead] = True

    s = copy_state(base)
    neutral_init = 0.1 * MAX_STRENGTH
    seat_init = 0.3 * MAX_STRENGTH
    s.owner = np.full(V, NEUTRAL, dtype=np.int32)
    s.strength = np.full(V, neutral_init, dtype=np.float32)
    if len(dead) > 0:
        s.owner[dead] = DEAD
        s.strength[dead] = 0.0
    for p, cell in enumerate(seats):
        c = int(cell)
        s.owner[c] = p
        s.strength[c] = seat_init
    return s, dead, pos3d


def _combine_actions(state, per_seat_actions: list[np.ndarray]) -> np.ndarray:
    N = state.N
    combined = np.full(N, ACTION_NOOP, dtype=np.int32)
    owner = state.owner
    for seat, actions in enumerate(per_seat_actions):
        mask = owner == seat
        combined[mask] = actions[mask]
    return combined


def _cells_per_seat(state, num_players: int) -> np.ndarray:
    return np.array(
        [int((state.owner == p).sum()) for p in range(num_players)], dtype=np.int64,
    )


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdiv", type=int, default=4,
                    help="icosahedron subdivision level. V = 10·4^subdiv + 2: "
                         "0→12, 1→42, 2→162, 3→642, 4→2562, 5→10242.")
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--dead-frac", type=float, default=0.40)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=10000)
    ap.add_argument("--record-stride", type=int, default=25)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--write-replay", action="store_true")
    ap.add_argument("--seats", type=str, default=None,
                    help=f"comma-separated solver names per seat "
                         f"({'/'.join(sorted(SOLVERS))}). Default: all 'lightning_attn'.")
    args = ap.parse_args()

    if args.seats:
        seat_solvers = [s.strip() for s in args.seats.split(",")]
        if len(seat_solvers) != args.num_players:
            raise SystemExit(
                f"--seats has {len(seat_solvers)} entries, expected {args.num_players}"
            )
        for s in seat_solvers:
            if s not in SOLVERS:
                raise SystemExit(f"unknown solver '{s}'. Choose from: {sorted(SOLVERS)}")
    else:
        seat_solvers = ["lightning_attn"] * args.num_players

    rng = np.random.default_rng(args.seed)
    print(f"v2 sphere solver: subdiv={args.subdiv} P={args.num_players} "
          f"max_ticks={args.max_ticks} dead_frac={args.dead_frac:.2f}")
    print(f"  seats: {seat_solvers}")

    t0 = time.time()
    state, dead, pos3d = _build_initial_state(
        args.subdiv, args.num_players, args.dead_frac, rng,
    )
    print(f"  V={state.N} live={int((state.owner != DEAD).sum())} "
          f"dead={len(dead)} (frac={len(dead)/state.N:.2f})")

    solver_fns = [SOLVERS[name] for name in seat_solvers]
    frames = [state_to_frame(state)]
    for t in range(1, args.max_ticks + 1):
        if t % args.ai_period_ticks == 0:
            per_seat = [solver_fns[seat](state, seat, rng=rng) for seat in range(args.num_players)]
            combined = _combine_actions(state, per_seat)
            state = apply_actions(state, combined)
        state = tick(state)
        if t % args.record_stride == 0:
            frames.append(state_to_frame(state))
        cells = _cells_per_seat(state, args.num_players)
        if (cells > 0).sum() <= 1:
            frames.append(state_to_frame(state))
            break

    dt = time.time() - t0
    cells = _cells_per_seat(state, args.num_players)
    dom = float(cells.max() / max(cells.sum(), 1))
    winner = int(cells.argmax()) if cells.max() > 0 and (cells > 0).sum() == 1 else -1
    if winner < 0:
        print(f"  stalemate at tick {state.tick} (dominance {dom:.2f}, alive {(cells > 0).sum()}, {dt:.1f}s)")
    else:
        print(f"  seat {winner} ({seat_solvers[winner]}) wins at tick {state.tick} "
              f"(dominance {dom:.2f}, {dt:.1f}s)")
    print(f"  recorded frames: {len(frames)}")

    if args.write_replay:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        tag = "+".join(sorted(set(seat_solvers)))
        # Cap file-tag length so a many-seat run doesn't generate a 250-char path.
        if len(tag) > 80:
            tag = tag[:77] + "_etc"
        name = f"sphere_v2_s{args.subdiv}_{tag}_{stamp}.flxr"
        path = DEFAULT_OUT_DIR / name
        # Sphere geometry shipped to the viewer via metadata.
        metadata = {
            "kind": "sphere_v2",
            "graph_kind": "sphere_icosphere_v1",
            "subdiv": args.subdiv,
            "model": f"sphere_{tag}",
            "ruleset": "v2-pressure",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "dead_cells": [int(x) for x in dead],
            "seats": seat_solvers,
            "iteration": 0,
            "generation": 0,
            # Geometry & topology, base64-encoded so the FLXR header stays small.
            "pos3d_b64": _b64(pos3d.astype(np.float32)),
            "neighbors_b64": _b64(state.neighbors.astype(np.int32)),
            "back_slot_b64": _b64(state.back_slot.astype(np.int32)),
        }
        # Header `radius` is meaningless on a sphere board; reuse the field to
        # carry the subdivision level so existing index displays still get a
        # number to show. The viewer keys off `graph_kind` to switch geometry.
        header = ReplayHeader(
            radius=args.subdiv, num_players=args.num_players,
            num_nodes=state.N, tick_stride=args.record_stride,
            dt_per_tick_ms=100, metadata=metadata,
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            w = ReplayWriter(f, header)
            for fr in frames:
                w.write_frame(fr)
            w.close()
        tmp.replace(path)
        append_index(DEFAULT_OUT_DIR, {
            "file": path.name,
            "saved_at": metadata["saved_at"],
            "kind": "sphere_v2", "model": metadata["model"],
            "ruleset": "v2-pressure",
            "seats": seat_solvers,
            "iteration": 0, "generation": 0,
            "radius": args.subdiv, "num_players": args.num_players,
            "graph_kind": "sphere_icosphere_v1",
        })
        print(f"  wrote replay: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
