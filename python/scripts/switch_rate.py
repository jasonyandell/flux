"""Switch-rate diagnostic for flux v2 solvers.

For each captured state, computes per-seat *modal target* — the enemy seat
receiving the largest fraction of that seat's active outflow slots. A
"switch" is a transition between two valid (≥0) modal targets in the
chronological series. Output is switches per 100 ticks per seat, aggregated
over self-play games.

Hypothesis under test (cf. wiki/topics/v2-temporal-strategy.md):
  - bfs: 0-2 switches / 100 ticks (geographic commitment by construction)
  - lightning_sum: 5-20 switches / 100 ticks (greedy on potential field,
    target flips as enemy regen re-orders the value of targets)

Usage:
    python scripts/switch_rate.py bfs lightning_sum lightning_sum_throttled \\
        --games 6 --radius 20 --num-players 6 --num-dead-cells 40 --max-ticks 4000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import K, apply_actions, tick
from flux_v2.state import State, copy_state
from scripts.run_v2_solver import (  # type: ignore
    SOLVERS,
    _build_initial_state,
    _cells_per_seat,
    _combine_actions,
)


def _modal_target_per_seat(state: State, num_players: int) -> np.ndarray:
    """For a single state, return (num_players,) int array of modal target
    enemy seat per source seat. -1 if seat has no active enemy-aimed
    outflows or no cells.
    """
    out = np.full(num_players, -1, dtype=np.int32)
    nb = state.neighbors
    owner = state.owner
    outflow = state.outflow
    for seat in range(num_players):
        mine = np.where(owner == seat)[0]
        if mine.size == 0:
            continue
        counts = np.zeros(num_players, dtype=np.int64)
        for c in mine:
            for k in range(K):
                if not outflow[c, k]:
                    continue
                d = int(nb[c, k])
                if d < 0:
                    continue
                od = int(owner[d])
                if od < 0 or od == seat or od >= num_players:
                    continue
                counts[od] += 1
        if counts.sum() > 0:
            out[seat] = int(counts.argmax())
    return out


def _switches_per_100_ticks(modal: np.ndarray, ticks_per_sample: int) -> np.ndarray:
    """(T, P) modal series sampled every `ticks_per_sample` ticks →
    (P,) switches per 100 ticks per seat. -1 → valid and valid → -1 don't
    count as switches. Time normalization uses the count of *transitions
    where both endpoints are valid*; the denominator is total ticks the
    seat had at least one valid target.
    """
    T, P = modal.shape
    out = np.zeros(P, dtype=np.float64)
    if T < 2:
        return out
    for p in range(P):
        prev = -1
        switches = 0
        valid_intervals = 0  # number of consecutive (cur valid, prev valid) sample-pairs
        for t in range(T):
            cur = int(modal[t, p])
            if cur >= 0:
                if prev >= 0:
                    valid_intervals += 1
                    if cur != prev:
                        switches += 1
                prev = cur
            else:
                prev = -1
        valid_ticks = valid_intervals * ticks_per_sample
        if valid_ticks > 0:
            out[p] = switches * 100.0 / valid_ticks
    return out


def run_game_with_diagnostic(
    radius: int,
    num_players: int,
    num_dead_cells: int,
    ai_period: int,
    max_ticks: int,
    rng: np.random.Generator,
    seat_solvers: list[str],
    sample_period_ticks: int,
    connect_mode: str = "retry",
):
    """Run one game, sampling modal-target state every `sample_period_ticks`
    ticks. Returns (final_state, modal_series, winner_seat). modal_series is
    (T, num_players) int.
    """
    state, _dead = _build_initial_state(
        radius, num_players, num_dead_cells, rng, connect_mode=connect_mode,
    )
    solver_fns = [SOLVERS[name] for name in seat_solvers]

    samples: list[np.ndarray] = [_modal_target_per_seat(state, num_players)]
    for t in range(1, max_ticks + 1):
        if t % ai_period == 0:
            per_seat = [solver_fns[seat](state, seat, rng=rng) for seat in range(num_players)]
            combined = _combine_actions(state, per_seat)
            state = apply_actions(state, combined)
        state = tick(state)
        if t % sample_period_ticks == 0:
            samples.append(_modal_target_per_seat(state, num_players))
        cells = _cells_per_seat(state, num_players)
        if (cells > 0).sum() <= 1:
            samples.append(_modal_target_per_seat(state, num_players))
            break

    cells = _cells_per_seat(state, num_players)
    winner = int(cells.argmax()) if cells.max() > 0 else -1
    modal = np.stack(samples, axis=0)
    return state, modal, winner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("solvers", nargs="+", type=str,
                    help="One or more solver names. Each runs as a self-play game.")
    ap.add_argument("--games", type=int, default=4,
                    help="Self-play games per solver.")
    ap.add_argument("--radius", type=int, default=20)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--num-dead-cells", type=int, default=40)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=4000)
    ap.add_argument("--sample-period-ticks", type=int, default=25,
                    help="How often (in ticks) to snapshot the modal-target state.")
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--connect-mode", choices=("retry", "carve"), default="retry")
    args = ap.parse_args()

    for name in args.solvers:
        if name not in SOLVERS:
            raise SystemExit(f"unknown solver '{name}'. Choose from: {sorted(SOLVERS)}")

    print(f"switch-rate diagnostic: solvers={args.solvers}")
    print(f"  config: R={args.radius} P={args.num_players} dead={args.num_dead_cells} "
          f"max_ticks={args.max_ticks} games/solver={args.games} "
          f"sample_period={args.sample_period_ticks} seed={args.seed}")
    print()

    rng_master = np.random.default_rng(args.seed)
    # Shared seed list across solvers so each solver sees the same boards.
    seeds = rng_master.integers(0, 2**31 - 1, size=args.games, dtype=np.int64)

    summary: dict[str, dict] = {}
    for name in args.solvers:
        per_game_rates: list[np.ndarray] = []
        per_game_alive: list[int] = []
        per_game_winner: list[int] = []
        per_game_ticks: list[int] = []
        for g, sd in enumerate(seeds):
            rng = np.random.default_rng(int(sd))
            seat_solvers = [name] * args.num_players
            t0 = time.time()
            state, modal, winner = run_game_with_diagnostic(
                args.radius, args.num_players, args.num_dead_cells,
                args.ai_period_ticks, args.max_ticks, rng,
                seat_solvers, args.sample_period_ticks,
                connect_mode=args.connect_mode,
            )
            rates = _switches_per_100_ticks(modal, args.sample_period_ticks)
            per_game_rates.append(rates)
            cells = _cells_per_seat(state, args.num_players)
            per_game_alive.append(int((cells > 0).sum()))
            per_game_winner.append(winner)
            per_game_ticks.append(state.tick)
            dt = time.time() - t0
            mean_rate = float(rates.mean())
            print(f"  [{name:>25s}] game {g}: switches/100t mean={mean_rate:.2f} "
                  f"per-seat={[f'{r:.1f}' for r in rates]}  "
                  f"winner={winner} ticks={state.tick} alive={(cells > 0).sum()}  ({dt:.1f}s)")
        all_rates = np.concatenate(per_game_rates)
        summary[name] = {
            "mean": float(all_rates.mean()),
            "median": float(np.median(all_rates)),
            "max": float(all_rates.max()),
            "stalemates": sum(1 for w in per_game_winner if w < 0),
            "mean_ticks": float(np.mean(per_game_ticks)),
        }

    print()
    print("== aggregate switches/100t (over all seats, all games) ==")
    width = max(len(n) for n in args.solvers)
    for name in args.solvers:
        s = summary[name]
        print(f"  {name:>{width}s}: mean={s['mean']:5.2f}  median={s['median']:5.2f}  "
              f"max={s['max']:5.1f}  stalemates={s['stalemates']}  "
              f"mean_ticks={s['mean_ticks']:.0f}")


if __name__ == "__main__":
    main()
