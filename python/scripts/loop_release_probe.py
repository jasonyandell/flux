"""Probe whether a charged friendly loop can beat direct v2 pressure.

The micro-scenario is intentionally tiny:

    B
   / \
  A - C      A, B, C are friendly max-strength cells.
  |          T is an enemy target adjacent to A.
  T

Direct baselines attack T immediately. Loop scenarios first hold the directed
cycle A->B->C->A, then release from A into T. This uses the normal
`apply_actions` API and the same JIT tick core as flux_v2.step, but passes
constants explicitly so EDGE_ALPHA, MAX_EDGE, REGEN_BASE_PER_TICK, and
MAX_STRENGTH can be swept without editing source constants.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flux_v2 import (  # noqa: E402
    ACTION_CLEAR_BASE,
    ACTION_NOOP,
    ACTION_SET_BASE,
    DEAD,
    HEX_DIRECTIONS,
    K,
    NEUTRAL,
    OPPOSITE_SLOT,
    State,
    apply_actions,
)
from flux_v2 import step as step_mod  # noqa: E402
from flux_v2.state import (  # noqa: E402
    WASTE_WEIGHT_CAP_BOUND,
    WASTE_WEIGHT_DEST_TERMINATED,
    WASTE_WEIGHT_NO_SPILL,
)

FRIEND = 0
ENEMY = 1
A = 0
B = 1
C = 2
T = 3


@dataclass(frozen=True)
class Params:
    edge_alpha: float
    max_edge: float
    regen: float
    max_strength: float
    target_strength: float
    charge_ticks: int
    release_horizon: int


@dataclass(frozen=True)
class Result:
    scenario: str
    release_mode: str
    params: Params
    pre_release_loop_pressure: float
    peak_attack_edge: float
    damage_10: float
    damage_25: float
    damage_50: float
    damage_end: float
    capture_after_release: int | None
    target_owner_end: int
    target_strength_end: float
    waste_total: float


def _parse_csv_numbers(raw: str, cast=float) -> list:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _state_for(max_strength: float, target_strength: float) -> State:
    coords = np.array(
        [
            (0, 0),   # A
            (1, 0),   # B
            (1, -1),  # C
            (-1, 0),  # T
        ],
        dtype=np.int32,
    )
    pos = np.zeros((len(coords), 2), dtype=np.float32)
    for i, (q, r) in enumerate(coords):
        pos[i] = (math.sqrt(3.0) * q + math.sqrt(3.0) * 0.5 * r, 1.5 * r)

    id_by_coord = {tuple(map(int, coord)): i for i, coord in enumerate(coords)}
    neighbors = np.full((len(coords), K), -1, dtype=np.int32)
    for i, (q, r) in enumerate(coords):
        for k, (dq, dr) in enumerate(HEX_DIRECTIONS):
            neighbors[i, k] = id_by_coord.get((int(q + dq), int(r + dr)), -1)

    owner = np.array([FRIEND, FRIEND, FRIEND, ENEMY], dtype=np.int32)
    strength = np.array(
        [max_strength, max_strength, max_strength, target_strength],
        dtype=np.float32,
    )
    return State(
        N=len(coords),
        pos=pos,
        coord=coords,
        neighbors=neighbors,
        owner=owner,
        strength=strength,
        outflow=np.zeros((len(coords), K), dtype=np.bool_),
        edge_pressure=np.zeros((len(coords), K), dtype=np.float32),
        tick=0,
        num_players=2,
    )


def _slot(state: State, src: int, dst: int) -> int:
    found = np.where(state.neighbors[src] == dst)[0]
    if len(found) != 1:
        raise ValueError(f"{src} does not have {dst} as a direct neighbor")
    return int(found[0])


def _actions_for(state: State, sets: Iterable[tuple[int, int]] = (), clears: Iterable[tuple[int, int]] = ()) -> np.ndarray:
    actions = np.full(state.N, ACTION_NOOP, dtype=np.int32)
    for src, dst in sets:
        actions[src] = ACTION_SET_BASE + _slot(state, src, dst)
    for src, dst in clears:
        actions[src] = ACTION_CLEAR_BASE + _slot(state, src, dst)
    return actions


def _tick_with_params(state: State, params: Params) -> State:
    s = step_mod.copy_state(state)
    new_owner, new_strength, new_outflow, new_edge_pressure, waste_delta = step_mod._tick_core(
        np.ascontiguousarray(s.owner, dtype=np.int32),
        np.ascontiguousarray(s.strength, dtype=np.float32),
        np.ascontiguousarray(s.outflow, dtype=np.bool_),
        np.ascontiguousarray(s.edge_pressure, dtype=np.float32),
        np.ascontiguousarray(s.neighbors, dtype=np.int32),
        np.asarray(OPPOSITE_SLOT, dtype=np.int32),
        max(s.num_players, 1),
        float(params.edge_alpha),
        float(params.max_strength),
        float(params.max_edge),
        float(params.regen),
        float(WASTE_WEIGHT_NO_SPILL),
        float(WASTE_WEIGHT_CAP_BOUND),
        float(WASTE_WEIGHT_DEST_TERMINATED),
        int(DEAD),
        int(NEUTRAL),
    )
    s.owner = new_owner
    s.strength = new_strength
    s.outflow = new_outflow
    s.edge_pressure = new_edge_pressure
    s.tick = state.tick + 1
    s.waste_total = state.waste_total + float(waste_delta)
    return s


def _run_ticks(state: State, params: Params, ticks: int) -> State:
    for _ in range(ticks):
        state = _tick_with_params(state, params)
    return state


def _attack_pressure_from_a(state: State) -> float:
    slot = _slot(state, A, T)
    return float(state.edge_pressure[A, slot])


def _loop_pressure(state: State) -> float:
    slots = [(_slot(state, A, B), A), (_slot(state, B, C), B), (_slot(state, C, A), C)]
    return float(np.mean([state.edge_pressure[src, slot] for slot, src in slots]))


def _record_damage(initial_target_strength: float, enemy_remaining: list[float], index: int) -> float:
    if not enemy_remaining:
        return 0.0
    capped_index = min(index, len(enemy_remaining) - 1)
    return float(initial_target_strength - enemy_remaining[capped_index])


def _simulate_release(
    state: State,
    params: Params,
    scenario: str,
    release_mode: str,
    after_tick: Callable[[int, State], State] | None = None,
) -> Result:
    initial_target_strength = float(state.strength[T])
    enemy_remaining: list[float] = []
    attack_edges: list[float] = []
    capture_after_release: int | None = None

    for t in range(params.release_horizon):
        state = _tick_with_params(state, params)
        remaining = 0.0 if int(state.owner[T]) == FRIEND else float(state.strength[T])
        enemy_remaining.append(remaining)
        attack_edges.append(_attack_pressure_from_a(state))
        if capture_after_release is None and int(state.owner[T]) == FRIEND:
            capture_after_release = t + 1
        if after_tick is not None:
            state = after_tick(t + 1, state)

    return Result(
        scenario=scenario,
        release_mode=release_mode,
        params=params,
        pre_release_loop_pressure=0.0,
        peak_attack_edge=max(attack_edges) if attack_edges else 0.0,
        damage_10=_record_damage(initial_target_strength, enemy_remaining, 9),
        damage_25=_record_damage(initial_target_strength, enemy_remaining, 24),
        damage_50=_record_damage(initial_target_strength, enemy_remaining, 49),
        damage_end=_record_damage(initial_target_strength, enemy_remaining, params.release_horizon - 1),
        capture_after_release=capture_after_release,
        target_owner_end=int(state.owner[T]),
        target_strength_end=float(state.strength[T]),
        waste_total=float(state.waste_total),
    )


def _direct_single(params: Params) -> Result:
    state = _state_for(params.max_strength, params.target_strength)
    state = apply_actions(state, _actions_for(state, sets=[(A, T)]))
    return _simulate_release(state, params, "direct_single", "immediate")


def _direct_feed(params: Params) -> Result:
    state = _state_for(params.max_strength, params.target_strength)
    state = apply_actions(state, _actions_for(state, sets=[(A, T), (B, A), (C, A)]))
    return _simulate_release(state, params, "direct_feed", "immediate")


def _charged_loop(params: Params, release_mode: str) -> Result:
    state = _state_for(params.max_strength, params.target_strength)
    state = apply_actions(state, _actions_for(state, sets=[(A, B), (B, C), (C, A)]))
    state = _run_ticks(state, params, params.charge_ticks)
    pre_release_loop_pressure = _loop_pressure(state)
    state = step_mod.copy_state(state)
    state.owner[T] = ENEMY
    state.strength[T] = np.float32(params.target_strength)

    if release_mode == "split":
        state = apply_actions(state, _actions_for(state, sets=[(A, T)]))
        result = _simulate_release(state, params, "charged_loop", release_mode)
    elif release_mode == "divert":
        state = apply_actions(state, _actions_for(state, sets=[(A, T)]))
        def clear_loop_after_first_tick(elapsed: int, current: State) -> State:
            if elapsed == 1:
                return apply_actions(current, _actions_for(current, clears=[(A, B)]))
            return current

        result = _simulate_release(
            state,
            params,
            "charged_loop",
            release_mode,
            after_tick=clear_loop_after_first_tick,
        )
    else:
        raise ValueError(f"unknown release mode: {release_mode}")
    return Result(
        scenario=result.scenario,
        release_mode=result.release_mode,
        params=result.params,
        pre_release_loop_pressure=pre_release_loop_pressure,
        peak_attack_edge=result.peak_attack_edge,
        damage_10=result.damage_10,
        damage_25=result.damage_25,
        damage_50=result.damage_50,
        damage_end=result.damage_end,
        capture_after_release=result.capture_after_release,
        target_owner_end=result.target_owner_end,
        target_strength_end=result.target_strength_end,
        waste_total=result.waste_total,
    )


def _run_sweep(args: argparse.Namespace) -> list[Result]:
    results: list[Result] = []
    for edge_alpha in args.edge_alpha:
        for max_edge in args.max_edge:
            for regen in args.regen:
                for max_strength in args.max_strength:
                    for target_strength in args.target_strength:
                        for charge_ticks in args.charge_ticks:
                            params = Params(
                                edge_alpha=edge_alpha,
                                max_edge=max_edge,
                                regen=regen,
                                max_strength=max_strength,
                                target_strength=target_strength,
                                charge_ticks=charge_ticks,
                                release_horizon=args.release_horizon,
                            )
                            if charge_ticks == args.charge_ticks[0]:
                                direct_params = Params(
                                    edge_alpha=edge_alpha,
                                    max_edge=max_edge,
                                    regen=regen,
                                    max_strength=max_strength,
                                    target_strength=target_strength,
                                    charge_ticks=0,
                                    release_horizon=args.release_horizon,
                                )
                                results.append(_direct_single(direct_params))
                                results.append(_direct_feed(direct_params))
                            for release_mode in args.release_mode:
                                results.append(_charged_loop(params, release_mode))
    return results


def _capture_text(tick: int | None) -> str:
    return "-" if tick is None else str(tick)


def _print_results(results: list[Result]) -> None:
    header = (
        "scenario,release,alpha,max_edge,regen,max_strength,target,charge,"
        "loop_pre,peak_attack,damage10,damage25,damage50,damage_end,capture,"
        "target_owner_end,target_strength_end,waste_total"
    )
    print(header)
    for r in results:
        p = r.params
        print(
            f"{r.scenario},{r.release_mode},{p.edge_alpha:g},{p.max_edge:g},{p.regen:g},"
            f"{p.max_strength:g},{p.target_strength:g},{p.charge_ticks},"
            f"{r.pre_release_loop_pressure:.2f},{r.peak_attack_edge:.2f},"
            f"{r.damage_10:.2f},{r.damage_25:.2f},{r.damage_50:.2f},"
            f"{r.damage_end:.2f},{_capture_text(r.capture_after_release)},"
            f"{r.target_owner_end},{r.target_strength_end:.2f},{r.waste_total:.2f}"
        )


def _print_summary(results: list[Result]) -> None:
    print()
    print("best_loop_cases")
    print("alpha,max_edge,regen,max_strength,target,charge,release,loop_damage50,best_direct_damage50,loop_capture,direct_capture")

    direct_by_key: dict[tuple[float, float, float, float, float], list[Result]] = {}
    loops: list[Result] = []
    for r in results:
        p = r.params
        key = (p.edge_alpha, p.max_edge, p.regen, p.max_strength, p.target_strength)
        if r.scenario.startswith("direct"):
            direct_by_key.setdefault(key, []).append(r)
        else:
            loops.append(r)

    scored: list[tuple[float, Result, Result | None]] = []
    for loop in loops:
        p = loop.params
        key = (p.edge_alpha, p.max_edge, p.regen, p.max_strength, p.target_strength)
        directs = direct_by_key.get(key, [])
        best_direct = max(directs, key=lambda r: r.damage_50, default=None)
        direct_damage = best_direct.damage_50 if best_direct else 0.0
        scored.append((loop.damage_50 - direct_damage, loop, best_direct))

    for _, loop, best_direct in sorted(scored, key=lambda row: row[0], reverse=True)[:12]:
        p = loop.params
        direct_damage = best_direct.damage_50 if best_direct else 0.0
        direct_capture = _capture_text(best_direct.capture_after_release if best_direct else None)
        print(
            f"{p.edge_alpha:g},{p.max_edge:g},{p.regen:g},{p.max_strength:g},"
            f"{p.target_strength:g},{p.charge_ticks},{loop.release_mode},{loop.damage_50:.2f},"
            f"{direct_damage:.2f},{_capture_text(loop.capture_after_release)},"
            f"{direct_capture}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-alpha", type=lambda s: _parse_csv_numbers(s, float), default=[1.0, 0.2, 0.05])
    parser.add_argument("--max-edge", type=lambda s: _parse_csv_numbers(s, float), default=[1000.0])
    parser.add_argument("--regen", type=lambda s: _parse_csv_numbers(s, float), default=[5.0, 10.0])
    parser.add_argument("--max-strength", type=lambda s: _parse_csv_numbers(s, float), default=[1000.0])
    parser.add_argument("--target-strength", type=lambda s: _parse_csv_numbers(s, float), default=[100.0, 250.0, 500.0])
    parser.add_argument("--charge-ticks", type=lambda s: _parse_csv_numbers(s, int), default=[0, 10, 25, 50, 100, 200])
    parser.add_argument("--release-horizon", type=int, default=60)
    parser.add_argument("--release-mode", type=lambda s: _parse_csv_numbers(s, str), default=["split", "divert"])
    args = parser.parse_args()

    results = _run_sweep(args)
    _print_results(results)
    _print_summary(results)


if __name__ == "__main__":
    main()
