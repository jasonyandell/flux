"""Run the old-school algorithmic solver on flux v2.

Plays G self-play games with every seat controlled by the same hand-written
heuristic (`flux_v2.solver.solver_actions`), prints per-game outcomes, and
writes one replay so the result is visible in the v2 displayer.

Usage:
    python scripts/run_v2_solver.py --radius 6 --num-players 6 --games 4

The solver doesn't read or write any model weights; this is a fixed-policy
baseline that the trained PPO policy should be able to beat once it's
working. Useful as a sanity check on game dynamics and as a reference
opponent.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import (
    ACTION_NOOP,
    DEAD,
    K,
    NEUTRAL,
    apply_actions,
    make_board,
    random_seat_and_dead,
    tick,
)
from flux_v2.replay import ReplayHeader, ReplayWriter, append_index, state_to_frame
from flux_v2.solver import solver_actions
from flux_v2.solver_lightning import lightning_solver_actions
from flux_v2.state import copy_state


def _lightning_sum(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="sum")


def _lightning_sum_pw(state, seat, rng=None):
    return lightning_solver_actions(state, seat, rng=rng, mode="sum_pw")


SOLVERS = {
    "bfs": solver_actions,
    "lightning": lightning_solver_actions,       # mode=max (original)
    "lightning_sum": _lightning_sum,             # value-iteration sum
    "lightning_sum_pw": _lightning_sum_pw,       # edge-pressure-weighted sum
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "public" / "v2" / "replays"


def _build_initial_state(
    radius: int,
    num_players: int,
    num_dead_cells: int,
    rng: np.random.Generator,
):
    """Build a single random board (random seats, connected dead-cell set)."""
    base = make_board(radius, num_players)
    seats, dead = random_seat_and_dead(
        base.N, num_players, num_dead_cells, rng,
        neighbors=base.neighbors, min_seat_dist=2, coord=base.coord,
    )
    s = copy_state(base)
    s.owner = np.full(base.N, NEUTRAL, dtype=np.int32)
    s.strength = np.full(base.N, 10.0, dtype=np.float32)
    if len(dead) > 0:
        s.owner[dead] = DEAD
        s.strength[dead] = 0.0
    for p, cell in enumerate(seats):
        c = int(cell)
        s.owner[c] = p
        s.strength[c] = 30.0
    return s, dead


def _combine_actions(state, per_seat_actions: list[np.ndarray]) -> np.ndarray:
    """Each seat returned a (N,) array of actions on its own cells (NOOP
    elsewhere). Merge into one (N,) by selecting each cell's current owner."""
    N = state.N
    combined = np.full(N, ACTION_NOOP, dtype=np.int32)
    owner = state.owner
    for seat, actions in enumerate(per_seat_actions):
        mask = owner == seat
        combined[mask] = actions[mask]
    return combined


def _cells_per_seat(state, num_players: int) -> np.ndarray:
    out = np.zeros(num_players, dtype=np.int64)
    for p in range(num_players):
        out[p] = int((state.owner == p).sum())
    return out


def run_game(
    radius: int,
    num_players: int,
    num_dead_cells: int,
    ai_period: int,
    max_ticks: int,
    rng: np.random.Generator,
    seat_solvers: list[str],
    record_stride: int = 25,
):
    """Run one game with per-seat solver assignment. Returns (final_state,
    frames, winner_seat, dead_cells)."""
    state, dead = _build_initial_state(radius, num_players, num_dead_cells, rng)
    solver_fns = [SOLVERS[name] for name in seat_solvers]

    frames = [state_to_frame(state)]
    for t in range(1, max_ticks + 1):
        if t % ai_period == 0:
            per_seat: list[np.ndarray] = []
            for seat in range(num_players):
                per_seat.append(solver_fns[seat](state, seat, rng=rng))
            combined = _combine_actions(state, per_seat)
            state = apply_actions(state, combined)
        state = tick(state)
        if t % record_stride == 0:
            frames.append(state_to_frame(state))

        # Early stop: at most one seat still has any cells.
        cells = _cells_per_seat(state, num_players)
        if (cells > 0).sum() <= 1:
            frames.append(state_to_frame(state))
            break

    cells = _cells_per_seat(state, num_players)
    winner = int(cells.argmax()) if cells.max() > 0 else -1
    return state, frames, winner, dead


def write_replay(
    path: Path,
    frames: list,
    radius: int,
    num_players: int,
    num_nodes: int,
    record_stride: int,
    metadata: dict,
) -> None:
    header = ReplayHeader(
        radius=radius, num_players=num_players, num_nodes=num_nodes,
        tick_stride=record_stride, dt_per_tick_ms=100,
        metadata=metadata,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        w = ReplayWriter(f, header)
        for fr in frames:
            w.write_frame(fr)
        w.close()
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--num-dead-cells", type=int, default=10)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=4000)
    ap.add_argument("--record-stride", type=int, default=25)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--write-replay", action="store_true",
                    help="write one .flxr to public/v2/replays/ for game 0")
    ap.add_argument("--seats", type=str, default=None,
                    help=f"comma-separated solver names per seat ({'/'.join(sorted(SOLVERS))}). "
                         f"Default: all 'bfs'. Example: bfs,lightning,bfs,lightning,bfs,lightning")
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
        seat_solvers = ["bfs"] * args.num_players

    rng = np.random.default_rng(args.seed)
    print(f"v2 solver play: radius={args.radius} P={args.num_players} "
          f"G={args.games} max_ticks={args.max_ticks} dead={args.num_dead_cells}")
    print(f"  seats: {seat_solvers}")

    win_counts = np.zeros(args.num_players, dtype=np.int64)
    stalemates = 0
    durations: list[int] = []

    first_game_frames = None
    first_game_dead = None
    first_game_state = None

    for g in range(args.games):
        t0 = time.time()
        state, frames, winner, dead = run_game(
            args.radius, args.num_players, args.num_dead_cells,
            args.ai_period_ticks, args.max_ticks, rng,
            seat_solvers, args.record_stride,
        )
        dt = time.time() - t0
        durations.append(state.tick)
        cells = _cells_per_seat(state, args.num_players)
        dom = float(cells.max() / max(cells.sum(), 1))
        if winner < 0 or (cells > 0).sum() > 1:
            stalemates += 1
            print(f"  game {g}: stalemate at tick {state.tick} "
                  f"(dominance {dom:.2f}, alive {(cells > 0).sum()}, {dt:.1f}s)")
        else:
            win_counts[winner] += 1
            print(f"  game {g}: seat {winner} wins at tick {state.tick} "
                  f"(dominance {dom:.2f}, {dt:.1f}s)")
        if g == 0:
            first_game_frames = frames
            first_game_dead = dead
            first_game_state = state

    print()
    print(f"  total: {args.games} games, mean ticks {np.mean(durations):.0f}")
    for p in range(args.num_players):
        print(f"    seat {p} ({seat_solvers[p]:>17s}): {int(win_counts[p])} wins")
    if stalemates:
        print(f"    stalemates: {stalemates}")
    # Aggregate by solver name.
    by_solver: dict[str, int] = {}
    for p, name in enumerate(seat_solvers):
        by_solver[name] = by_solver.get(name, 0) + int(win_counts[p])
    if len(set(seat_solvers)) > 1:
        print("  by solver:")
        for name, w in sorted(by_solver.items(), key=lambda kv: -kv[1]):
            seats_count = seat_solvers.count(name)
            print(f"    {name:>17s} ({seats_count} seats): {w} wins")

    if args.write_replay and first_game_frames is not None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        tag = "+".join(sorted(set(seat_solvers)))
        name = f"solver_v2_{tag}_{stamp}.flxr"
        path = DEFAULT_OUT_DIR / name
        dead_list = [int(x) for x in first_game_dead] if first_game_dead is not None else []
        tag = "+".join(sorted(set(seat_solvers)))
        metadata = {
            "kind": "solver_v2",
            "model": f"solver_{tag}",
            "ruleset": "v2-pressure",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "dead_cells": dead_list,
            "seats": seat_solvers,
            "iteration": 0,
            "generation": 0,
        }
        write_replay(
            path, first_game_frames,
            args.radius, args.num_players, first_game_state.N,
            args.record_stride, metadata,
        )
        append_index(DEFAULT_OUT_DIR, {
            "file": path.name,
            "saved_at": metadata["saved_at"],
            "kind": "solver_v2", "model": metadata["model"],
            "ruleset": "v2-pressure",
            "seats": seat_solvers,
            "iteration": 0, "generation": 0,
            "radius": args.radius, "num_players": args.num_players,
        })
        print(f"  wrote replay: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
