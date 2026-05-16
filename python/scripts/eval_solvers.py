"""Matched-pair tournament between two flux v2 solvers.

For each seed, runs two games on the *same* random board:
  - game A: solver_A occupies even seats, solver_B occupies odd seats.
  - game B: solver_B occupies even seats, solver_A occupies odd seats.

Pairs are "coherent" when one solver wins both halves (a real signal,
unaffected by seat bias). Pairs where each solver wins one half are
ambiguous — typically half are A-coherent, half B-coherent under noise.

Output: per-solver win counts (raw + by seat parity), coherent-pair sign
test result with Wilson 95% CI, mean game duration, stalemate rate, and
average switch-rate per seat (if --diagnostic). Reuses run_game from
run_v2_solver.py so board generation and step semantics stay identical
to the baseline runner.

Usage:
    python scripts/eval_solvers.py bfs lightning_sum \\
        --pairs 50 --radius 20 --num-players 6 --num-dead-cells 40 --max-ticks 6000

Methodology reference: wiki/topics/v2-overnight-research.md exp 22-24.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2.state import K
from scripts.run_v2_solver import (  # type: ignore
    SOLVERS,
    _cells_per_seat,
    run_game,
)

from flux_v2.state import State


def _modal_target_series(states: list[State], num_players: int) -> np.ndarray:
    """For each captured state (assumed in chronological order), return a
    (T, num_players) int array of each seat's modal target seat — the enemy
    seat receiving the most outgoing pressure from that seat right now.

    Modal target = argmax over enemy seats e of:
        sum over (c, k) with owner[c]==seat and outflow[c,k]==True and
        owner[nb[c,k]]==e of 1

    Returns -1 for seats that have no active enemy-aimed outflows (no target).
    """
    T = len(states)
    out = np.full((T, num_players), -1, dtype=np.int32)
    if T == 0:
        return out
    for t, s in enumerate(states):
        nb = s.neighbors
        owner = s.owner
        outflow = s.outflow
        for seat in range(num_players):
            mine_cells = np.where(owner == seat)[0]
            if mine_cells.size == 0:
                continue
            # count[e] = # of active enemy-aimed outflow slots owned by `seat`
            #            whose destination is seat e.
            counts = np.zeros(num_players, dtype=np.int64)
            for c in mine_cells:
                for k in range(K):
                    if not outflow[c, k]:
                        continue
                    d = int(nb[c, k])
                    if d < 0:
                        continue
                    od = int(owner[d])
                    if od < 0 or od == seat:
                        continue
                    if od >= num_players:
                        continue
                    counts[od] += 1
            if counts.sum() == 0:
                continue
            out[t, seat] = int(counts.argmax())
    return out


def _switches_per_100_ticks(modal: np.ndarray, ticks_per_sample: int) -> np.ndarray:
    """Given (T, P) modal-target series sampled every `ticks_per_sample` ticks,
    return (P,) switches per 100 ticks per seat.

    A switch is a transition between two valid (≥0) modal targets. -1 → x and
    x → -1 don't count (seat had no target then acquired one, or vice versa).
    """
    T, P = modal.shape
    out = np.zeros(P, dtype=np.float64)
    if T < 2:
        return out
    total_ticks = (T - 1) * ticks_per_sample
    if total_ticks <= 0:
        return out
    for p in range(P):
        prev = -1
        switches = 0
        for t in range(T):
            cur = int(modal[t, p])
            if cur < 0:
                prev = -1
                continue
            if prev >= 0 and cur != prev:
                switches += 1
            prev = cur
        out[p] = switches * 100.0 / total_ticks
    return out


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for proportion. Returns (point, lo, hi)."""
    if n == 0:
        return 0.5, 0.0, 1.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, center - radius, center + radius


def _assign_seats(num_players: int, name_even: str, name_odd: str) -> list[str]:
    """Alternate solvers across seats."""
    return [name_even if (s % 2 == 0) else name_odd for s in range(num_players)]


def _winner_solver(seats: list[str], winner_seat: int) -> str | None:
    if winner_seat < 0:
        return None
    return seats[winner_seat]


def run_pair(
    a: str,
    b: str,
    radius: int,
    num_players: int,
    num_dead_cells: int,
    ai_period: int,
    max_ticks: int,
    seed: int,
    connect_mode: str,
    capture_diagnostic: bool,
    diagnostic_period_ticks: int,
) -> dict:
    """Run two games on the same random seed: A-even/B-odd, then B-even/A-odd.
    Returns a dict with the two outcomes and optional per-game diagnostic.
    """
    seats_g1 = _assign_seats(num_players, a, b)
    seats_g2 = _assign_seats(num_players, b, a)

    rng_g1 = np.random.default_rng(seed)
    rng_g2 = np.random.default_rng(seed)  # same seed → same board

    capture_states_every = diagnostic_period_ticks if capture_diagnostic else None

    s1, frames1, winner1, _ = run_game(
        radius, num_players, num_dead_cells, ai_period, max_ticks, rng_g1,
        seats_g1, record_stride=10**9, connect_mode=connect_mode,
    )
    s2, frames2, winner2, _ = run_game(
        radius, num_players, num_dead_cells, ai_period, max_ticks, rng_g2,
        seats_g2, record_stride=10**9, connect_mode=connect_mode,
    )

    w1 = _winner_solver(seats_g1, winner1)
    w2 = _winner_solver(seats_g2, winner2)

    return {
        "seed": seed,
        "g1_seats": seats_g1, "g1_winner_seat": winner1, "g1_winner_solver": w1,
        "g1_ticks": s1.tick, "g1_cells": _cells_per_seat(s1, num_players).tolist(),
        "g2_seats": seats_g2, "g2_winner_seat": winner2, "g2_winner_solver": w2,
        "g2_ticks": s2.tick, "g2_cells": _cells_per_seat(s2, num_players).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("solver_a", type=str)
    ap.add_argument("solver_b", type=str)
    ap.add_argument("--pairs", type=int, default=30,
                    help="Number of matched pairs. Total games = 2 * pairs.")
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--num-dead-cells", type=int, default=40)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--connect-mode", choices=("retry", "carve"), default="retry")
    args = ap.parse_args()

    for name in (args.solver_a, args.solver_b):
        if name not in SOLVERS:
            raise SystemExit(f"unknown solver '{name}'. Choose from: {sorted(SOLVERS)}")

    print(f"matched-pair eval: {args.solver_a} vs {args.solver_b}")
    print(f"  config: R={args.radius} P={args.num_players} dead={args.num_dead_cells} "
          f"max_ticks={args.max_ticks} pairs={args.pairs} seed={args.seed}")
    print()

    rng_seed_master = np.random.default_rng(args.seed)
    seeds = rng_seed_master.integers(0, 2**31 - 1, size=args.pairs, dtype=np.int64)

    wins_by_solver = {args.solver_a: 0, args.solver_b: 0}
    wins_by_solver_even = {args.solver_a: 0, args.solver_b: 0}
    wins_by_solver_odd = {args.solver_a: 0, args.solver_b: 0}
    stalemates = 0
    durations: list[int] = []
    coherent_a = 0  # pairs where A wins both halves
    coherent_b = 0  # pairs where B wins both halves
    split = 0       # 1-1
    unresolved = 0  # at least one half stalemated

    t_start = time.time()
    for i, sd in enumerate(seeds):
        r = run_pair(
            args.solver_a, args.solver_b,
            args.radius, args.num_players, args.num_dead_cells,
            args.ai_period_ticks, args.max_ticks, int(sd),
            args.connect_mode,
            capture_diagnostic=False, diagnostic_period_ticks=100,
        )
        # Tally
        for half in ("g1", "g2"):
            seats = r[f"{half}_seats"]
            w_seat = r[f"{half}_winner_seat"]
            ticks = r[f"{half}_ticks"]
            durations.append(ticks)
            if w_seat < 0:
                stalemates += 1
                continue
            w = seats[w_seat]
            wins_by_solver[w] += 1
            if w_seat % 2 == 0:
                wins_by_solver_even[w] += 1
            else:
                wins_by_solver_odd[w] += 1

        w1 = r["g1_winner_solver"]
        w2 = r["g2_winner_solver"]
        if w1 is None or w2 is None:
            unresolved += 1
        elif w1 == w2 == args.solver_a:
            coherent_a += 1
        elif w1 == w2 == args.solver_b:
            coherent_b += 1
        else:
            split += 1
        if (i + 1) % 10 == 0 or i + 1 == args.pairs:
            elapsed = time.time() - t_start
            print(f"  [{i+1:4d}/{args.pairs}] coherent: A={coherent_a} B={coherent_b} "
                  f"split={split} unresolved={unresolved} ({elapsed:.0f}s)")

    print()
    total_games = 2 * args.pairs
    n_decided = total_games - stalemates
    print(f"== raw win counts (over {total_games} games, {stalemates} stalemates) ==")
    for name in (args.solver_a, args.solver_b):
        w = wins_by_solver[name]
        pct = 100.0 * w / max(n_decided, 1)
        print(f"  {name:>25s}: {w:3d}  ({pct:5.1f}% of decided)")
    print()
    print("== by seat parity (seat-bias check) ==")
    for name in (args.solver_a, args.solver_b):
        print(f"  {name:>25s}: even seats {wins_by_solver_even[name]:3d}  "
              f"odd seats {wins_by_solver_odd[name]:3d}")
    print()
    print("== matched-pair coherent results ==")
    n_coherent = coherent_a + coherent_b
    if n_coherent == 0:
        print("  no coherent pairs — increase --pairs or check that solvers actually differ")
    else:
        pa, lo_a, hi_a = _wilson_ci(coherent_a, n_coherent)
        pb, lo_b, hi_b = _wilson_ci(coherent_b, n_coherent)
        print(f"  coherent pairs: {n_coherent} / {args.pairs}  "
              f"(split: {split}, unresolved: {unresolved})")
        print(f"    {args.solver_a:>25s}: {coherent_a:3d} ({pa:.2%})  "
              f"95% CI [{lo_a:.2%}, {hi_a:.2%}]")
        print(f"    {args.solver_b:>25s}: {coherent_b:3d} ({pb:.2%})  "
              f"95% CI [{lo_b:.2%}, {hi_b:.2%}]")
        # Sign test: H0 = each solver equally likely to win a coherent pair.
        # Two-sided p ≈ binomial tail.
        p_signtest = 0.0
        n = n_coherent
        k_extreme = max(coherent_a, coherent_b)
        for k in range(k_extreme, n + 1):
            p_signtest += math.comb(n, k) * (0.5 ** n)
        p_signtest *= 2  # two-sided
        p_signtest = min(p_signtest, 1.0)
        leader = args.solver_a if coherent_a > coherent_b else args.solver_b
        signed_adv_pp = (coherent_a - coherent_b) / n_coherent * 100
        print(f"    sign test p ≈ {p_signtest:.4f}  "
              f"(leader: {leader}, signed advantage {signed_adv_pp:+.1f}pp coherent)")
    print()
    if durations:
        print(f"  mean game duration: {np.mean(durations):.0f} ticks "
              f"(median {np.median(durations):.0f}, max {np.max(durations)})")
    print(f"  wallclock: {time.time() - t_start:.0f}s "
          f"({(time.time() - t_start) / max(total_games, 1):.1f}s/game)")


if __name__ == "__main__":
    main()
