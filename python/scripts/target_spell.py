"""Target-spell diagnostic for flux v2 solver policies.

A "spell" is a consecutive run where a seat's modal enemy target stays the
same. This is a coarser diagnostic than per-AI-tick switch rate: it asks
whether a policy keeps pressure on a target until that target dies, or
abandons it while the target is still alive.

Usage:
    python scripts/target_spell.py lightning_sum_throttled \\
        --games 8 --radius 25 --num-players 12 --num-dead-cells 200

    python scripts/target_spell.py bfs lightning_sum lightning_sum_throttled \\
        --repeat --games 4
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import ACTION_NOOP, K, apply_actions, tick
from flux_v2.state import State
from scripts.run_v2_solver import (  # type: ignore
    SOLVERS,
    _build_initial_state,
    _cells_per_seat,
    _combine_actions,
    _solver_instance,
)


@dataclass
class Spell:
    solver: str
    game: int
    seat: int
    target: int
    start_tick: int
    end_tick: int
    target_dead: bool
    abandoned: bool

    @property
    def duration(self) -> int:
        return max(0, self.end_tick - self.start_tick)


def _modal_target_per_seat(state: State, num_players: int) -> np.ndarray:
    """Return modal enemy target per seat, or -1 when no enemy-aimed outflow."""
    out = np.full(num_players, -1, dtype=np.int32)
    nb = state.neighbors
    owner = state.owner
    outflow = state.outflow
    for seat in range(num_players):
        counts = np.zeros(num_players, dtype=np.float32)
        mine = np.where(owner == seat)[0]
        if mine.size == 0:
            continue
        for c in mine:
            for k in range(K):
                if not outflow[c, k]:
                    continue
                d = int(nb[c, k])
                if d < 0:
                    continue
                od = int(owner[d])
                if od >= 0 and od != seat and od < num_players:
                    counts[od] += 1.0
        if counts.sum() > 0.0:
            out[seat] = int(counts.argmax())
    return out


def _alive_by_seat(state: State, num_players: int) -> np.ndarray:
    cells = _cells_per_seat(state, num_players)
    return cells > 0


def _seat_solver_names(names: list[str], num_players: int, repeat: bool) -> list[str]:
    if len(names) == 1:
        return names * num_players
    if repeat:
        return [names[i % len(names)] for i in range(num_players)]
    if len(names) != num_players:
        raise SystemExit(
            f"provide exactly {num_players} solvers, one solver, or pass --repeat"
        )
    return names


def _close_spell(
    spells: list[Spell],
    solver_name: str,
    game_idx: int,
    seat: int,
    target: int,
    start_tick: int,
    end_tick: int,
    alive: np.ndarray,
    changed_target: bool,
    min_duration: int,
) -> None:
    duration = max(0, end_tick - start_tick)
    if target < 0 or duration < min_duration:
        return
    target_dead = not bool(alive[target])
    abandoned = changed_target and not target_dead
    spells.append(
        Spell(
            solver=solver_name,
            game=game_idx,
            seat=seat,
            target=target,
            start_tick=start_tick,
            end_tick=end_tick,
            target_dead=target_dead,
            abandoned=abandoned,
        )
    )


def run_game_with_spells(
    *,
    game_idx: int,
    radius: int,
    num_players: int,
    num_dead_cells: int,
    ai_period_ticks: int,
    max_ticks: int,
    sample_period_ticks: int,
    min_spell_ticks: int,
    rng: np.random.Generator,
    seat_solvers: list[str],
    connect_mode: str,
) -> tuple[State, int, list[Spell]]:
    state, _dead = _build_initial_state(
        radius, num_players, num_dead_cells, rng, connect_mode=connect_mode,
    )
    solver_fns = [_solver_instance(name) for name in seat_solvers]
    spells: list[Spell] = []

    cur_target = np.full(num_players, -1, dtype=np.int32)
    spell_start = np.zeros(num_players, dtype=np.int32)
    last_sample_tick = 0

    def sample(tick_now: int) -> None:
        nonlocal cur_target, spell_start, last_sample_tick
        modal = _modal_target_per_seat(state, num_players)
        alive = _alive_by_seat(state, num_players)
        for seat in range(num_players):
            old = int(cur_target[seat])
            new = int(modal[seat])
            if old == new:
                continue
            _close_spell(
                spells,
                seat_solvers[seat],
                game_idx,
                seat,
                old,
                int(spell_start[seat]),
                tick_now,
                alive,
                changed_target=(new >= 0),
                min_duration=min_spell_ticks,
            )
            cur_target[seat] = new
            spell_start[seat] = tick_now
        last_sample_tick = tick_now

    sample(0)
    for t in range(1, max_ticks + 1):
        if t % ai_period_ticks == 0:
            per_seat = [
                solver_fns[seat](state, seat, rng=rng)
                for seat in range(num_players)
            ]
            combined = _combine_actions(state, per_seat)
            state = apply_actions(state, combined)
        state = tick(state)
        if t % sample_period_ticks == 0:
            sample(t)
        cells = _cells_per_seat(state, num_players)
        if (cells > 0).sum() <= 1:
            sample(t)
            break

    alive = _alive_by_seat(state, num_players)
    for seat in range(num_players):
        _close_spell(
            spells,
            seat_solvers[seat],
            game_idx,
            seat,
            int(cur_target[seat]),
            int(spell_start[seat]),
            int(state.tick if state.tick > 0 else last_sample_tick),
            alive,
            changed_target=False,
            min_duration=min_spell_ticks,
        )
    cells = _cells_per_seat(state, num_players)
    winner = int(cells.argmax()) if cells.max() > 0 else -1
    return state, winner, spells


def _summarize(spells: list[Spell]) -> None:
    by_solver: dict[str, list[Spell]] = {}
    for sp in spells:
        by_solver.setdefault(sp.solver, []).append(sp)
    print("== target spell summary ==")
    for solver in sorted(by_solver):
        ss = by_solver[solver]
        n = len(ss)
        if n == 0:
            continue
        completed = sum(1 for s in ss if s.target_dead)
        abandoned = sum(1 for s in ss if s.abandoned)
        durations = np.array([s.duration for s in ss], dtype=np.float64)
        print(
            f"  {solver:>25s}: spells={n:4d}  "
            f"complete={completed/n:6.1%}  abandon={abandoned/n:6.1%}  "
            f"mean_t={durations.mean():7.1f}  p50_t={np.median(durations):7.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("solvers", nargs="+", choices=tuple(SOLVERS))
    ap.add_argument("--repeat", action="store_true",
                    help="Repeat the provided solver list across seats.")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--radius", type=int, default=25)
    ap.add_argument("--num-players", type=int, default=12)
    ap.add_argument("--num-dead-cells", type=int, default=200)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=12000)
    ap.add_argument("--sample-period-ticks", type=int, default=250)
    ap.add_argument("--min-spell-ticks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--connect-mode", choices=("retry", "carve"), default="retry")
    args = ap.parse_args()

    seat_solvers = _seat_solver_names(args.solvers, args.num_players, args.repeat)
    print("target-spell diagnostic")
    print(
        f"  seats={seat_solvers} R={args.radius} P={args.num_players} "
        f"dead={args.num_dead_cells} games={args.games} seed={args.seed} "
        f"sample={args.sample_period_ticks} min_spell={args.min_spell_ticks}"
    )
    rng_master = np.random.default_rng(args.seed)
    seeds = rng_master.integers(0, 2**31 - 1, size=args.games, dtype=np.int64)
    all_spells: list[Spell] = []
    for g, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        t0 = time.time()
        state, winner, spells = run_game_with_spells(
            game_idx=g,
            radius=args.radius,
            num_players=args.num_players,
            num_dead_cells=args.num_dead_cells,
            ai_period_ticks=args.ai_period_ticks,
            max_ticks=args.max_ticks,
            sample_period_ticks=args.sample_period_ticks,
            min_spell_ticks=args.min_spell_ticks,
            rng=rng,
            seat_solvers=seat_solvers,
            connect_mode=args.connect_mode,
        )
        all_spells.extend(spells)
        alive = int((_cells_per_seat(state, args.num_players) > 0).sum())
        print(
            f"  game {g}: winner={winner} alive={alive} ticks={state.tick} "
            f"spells={len(spells)} ({time.time() - t0:.1f}s)"
        )
    _summarize(all_spells)


if __name__ == "__main__":
    main()
