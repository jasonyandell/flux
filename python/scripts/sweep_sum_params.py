"""Sweep over lightning_sum hyperparameters vs a default-sum opponent.

Same harness shape as sweep_attn_params.py, but the variant being tested is
sum-mode with different (gamma, weak_bonus, expand_bonus). The opponent is
plain lightning_sum with defaults. Tests whether any sum-tuning beats the
default sum (which is the dominant solver under big-bag rules at R=20).
"""
from __future__ import annotations

import argparse
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


def build_state(radius, num_players, num_dead, rng, max_path_mult=4):
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


def make_sum(kw):
    def fn(state, seat, rng=None):
        return lightning_solver_actions(state, seat, rng=rng, mode="sum", **kw)
    return fn


def evaluate(kw, games, radius, dead, seed):
    rng = np.random.default_rng(seed)
    variant_fn = make_sum(kw)
    default_fn = make_sum({})
    seat_fns = [variant_fn, default_fn, variant_fn, default_fn, variant_fn, default_fn]
    wins = {"variant": 0, "default": 0, "stale": 0}
    durations = []
    for g in range(games):
        state = build_state(radius, 6, dead, rng)
        winner = play_game(state, seat_fns, ai_period=5, max_ticks=10000, rng=rng)
        durations.append(int(state.tick))
        if winner < 0:
            wins["stale"] += 1
        elif winner in (0, 2, 4):
            wins["variant"] += 1
        else:
            wins["default"] += 1
    return wins, float(np.mean(durations))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--num-dead", type=int, default=126)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFF)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    configs = [
        {"name": "default(baseline)", "kw": {}},
        # gamma sweep
        {"name": "gamma=0.7", "kw": {"gamma": 0.7}},
        {"name": "gamma=0.92", "kw": {"gamma": 0.92}},
        {"name": "gamma=0.97", "kw": {"gamma": 0.97}},
        # weak_bonus sweep (focus on weakest enemies)
        {"name": "weak_bonus=2.0", "kw": {"weak_bonus": 2.0}},
        {"name": "weak_bonus=5.0", "kw": {"weak_bonus": 5.0}},
        {"name": "weak_bonus=0.5", "kw": {"weak_bonus": 0.5}},
        # expand_bonus sweep (focus on neutrals)
        {"name": "expand=0.1", "kw": {"expand_bonus": 0.1}},
        {"name": "expand=1.0", "kw": {"expand_bonus": 1.0}},
        {"name": "expand=1.5", "kw": {"expand_bonus": 1.5}},
        # combos
        {"name": "focused_attack", "kw": {"weak_bonus": 3.0, "expand_bonus": 0.2}},
        {"name": "land_grab", "kw": {"weak_bonus": 0.5, "expand_bonus": 1.5}},
        {"name": "long_field", "kw": {"gamma": 0.95, "weak_bonus": 2.0}},
    ]

    print(f"sum sweep: R={args.radius} dead={args.num_dead} games={args.games} seed={args.seed}")
    results = []
    for cfg in configs:
        t0 = time.time()
        wins, dur = evaluate(cfg["kw"], args.games, args.radius, args.num_dead, args.seed)
        dt = time.time() - t0
        share = wins["variant"] / max(args.games, 1)
        print(f"  {cfg['name']:>20s}: variant={wins['variant']:>2d}  "
              f"def={wins['default']:>2d}  stale={wins['stale']:>2d}  "
              f"share={share:.0%}  ticks={dur:.0f}  ({dt:.1f}s)", flush=True)
        results.append({**cfg, **wins, "share": share, "ticks": dur})
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
