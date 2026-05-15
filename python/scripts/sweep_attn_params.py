"""Hyperparameter sweep for lightning_attn variants vs lightning_sum.

Direct harness — bypasses run_v2_solver.py's SOLVERS registry so we can call
lightning_solver_actions with arbitrary kwargs per seat. Reports per-config
win rate against lightning_sum across N games.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import ACTION_NOOP, DEAD, NEUTRAL, apply_actions, make_board, tick
from flux_v2.graph import (
    max_seat_pair_distance,
    random_seat_and_dead,
    seats_mutually_reachable,
)
from flux_v2.solver_lightning import lightning_solver_actions
from flux_v2.state import MAX_STRENGTH, copy_state


def build_state(radius: int, num_players: int, num_dead: int, rng, max_path_mult: int = 4):
    base = make_board(radius, num_players)
    max_path = max(max_path_mult * radius, 6)
    for _ in range(200):
        seats, dead = random_seat_and_dead(
            base.N, num_players, num_dead, rng,
            neighbors=base.neighbors, min_seat_dist=2, coord=base.coord,
        )
        if not seats_mutually_reachable(seats, dead, base.neighbors):
            continue
        worst = max_seat_pair_distance(seats, dead, base.neighbors)
        if 0 <= worst <= max_path:
            break
    else:
        raise RuntimeError("board sampler failed")
    s = copy_state(base)
    s.owner = np.full(base.N, NEUTRAL, dtype=np.int32)
    s.strength = np.full(base.N, 0.1 * MAX_STRENGTH, dtype=np.float32)
    if len(dead) > 0:
        s.owner[dead] = DEAD
        s.strength[dead] = 0.0
    for p, cell in enumerate(seats):
        c = int(cell)
        s.owner[c] = p
        s.strength[c] = 0.3 * MAX_STRENGTH
    return s


def play_game(state, seat_fns, ai_period, max_ticks, rng):
    """Run until 1 seat alive or max_ticks. Returns winning seat or -1."""
    P = state.num_players
    for t in range(1, max_ticks + 1):
        if t % ai_period == 0:
            per_seat = [seat_fns[s](state, s, rng) for s in range(P)]
            combined = np.full(state.N, ACTION_NOOP, dtype=np.int32)
            for s in range(P):
                m = state.owner == s
                combined[m] = per_seat[s][m]
            state = apply_actions(state, combined)
        state = tick(state)
        cells = [int((state.owner == p).sum()) for p in range(P)]
        if sum(1 for c in cells if c > 0) <= 1:
            return int(np.argmax(cells)) if max(cells) > 0 else -1
    return -1


def make_attn(kw: dict):
    """Return a solver_fn(state, seat, rng) bound to attn mode with overrides."""
    def fn(state, seat, rng=None):
        return lightning_solver_actions(state, seat, rng=rng, mode="attn", **kw)
    return fn


def make_sum():
    def fn(state, seat, rng=None):
        return lightning_solver_actions(state, seat, rng=rng, mode="sum")
    return fn


def evaluate_config(kw: dict, games: int, radius: int, dead: int, seed: int):
    """Run `games` head-to-head, alternating 3 attn-variant seats vs 3 sum.
    Returns dict with win counts."""
    rng = np.random.default_rng(seed)
    attn_fn = make_attn(kw)
    sum_fn = make_sum()
    seat_fns = [attn_fn, sum_fn, attn_fn, sum_fn, attn_fn, sum_fn]

    wins = {"attn_variant": 0, "sum": 0, "stalemate": 0}
    durations = []
    for g in range(games):
        state = build_state(radius, 6, dead, rng)
        winner = play_game(state, seat_fns, ai_period=5, max_ticks=10000, rng=rng)
        durations.append(state.tick)
        if winner < 0:
            wins["stalemate"] += 1
        elif winner in (0, 2, 4):
            wins["attn_variant"] += 1
        else:
            wins["sum"] += 1
    return wins, float(np.mean(durations))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--num-dead", type=int, default=126)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFF)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    # Sweep space — small Cartesian product.
    configs = []
    # Baseline (defaults)
    configs.append({"name": "defaults", "kw": {}})
    # Sweep deep_threshold (controls α ramp depth)
    for dt in (1.0, 3.0, 5.0):
        configs.append({"name": f"deep_thresh={dt}", "kw": {"deep_threshold": dt}})
    # Sweep gamma (field decay)
    for g in (0.7, 0.92, 0.95):
        configs.append({"name": f"gamma={g}", "kw": {"gamma": g}})
    # Sweep build-release
    for br in (0.5, 0.7, 0.85, 0.95):
        configs.append({"name": f"build_release={br}", "kw": {"build_release_frac": br}})
    # Crazy: very tight relay threshold
    configs.append({"name": "tight_relay", "kw": {"relay_thresh": 0.8}})
    # Crazy: huge expand_bonus
    configs.append({"name": "big_expand", "kw": {"expand_bonus": 1.5}})

    print(f"sweep: R={args.radius} dead={args.num_dead} games={args.games} seed={args.seed}")
    print(f"  {len(configs)} configs")

    results = []
    for cfg in configs:
        t0 = time.time()
        wins, mean_dur = evaluate_config(
            cfg["kw"], args.games, args.radius, args.num_dead, args.seed,
        )
        dt = time.time() - t0
        win_share = wins["attn_variant"] / max(args.games, 1)
        line = (f"  {cfg['name']:>22s}: attn={wins['attn_variant']:>2d}  "
                f"sum={wins['sum']:>2d}  stale={wins['stalemate']:>2d}  "
                f"attn_share={win_share:.0%}  mean_ticks={mean_dur:.0f}  ({dt:.1f}s)")
        print(line)
        results.append({**cfg, **wins, "win_share": win_share, "mean_ticks": mean_dur})

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"wrote results to {args.output}")


if __name__ == "__main__":
    main()
