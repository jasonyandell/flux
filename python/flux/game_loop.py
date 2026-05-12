"""MLX game runner: plays G games in parallel on the GPU.

Public surface:
    play_batch_games(rosters_list, config, record_game_idx=None) → list[GameResult]
    play_game(roster, config, record=None) → GameResult   # thin G=1 wrapper

State design:
    - Owner / strength held as (G, N) MLX tensors. Step + NN run batched.
    - Flow lists held per-game in Python (cheap reconcile; differs across games).
    - When games "win" (≤1 player alive), they freeze via live_mask. Frozen
      games' state is preserved and they don't get NN-inferred or stepped.
    - Loop exits when all games are frozen OR config.ticks is reached.

Per-seat fitness mirrors the JS evolution code:
    end_cells + early_weight * mid_cells - linger_penalty * opp_cells_at_end
where end_cells / mid_cells are captured at the game's *decision point* (the
tick it first hit ≤1-player-alive), or at the tick cap if it never did. This
naturally rewards quick wins without an explicit "tick spent" term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .genome import build_neighbor_table
from .graph import make_initial_state
from .mlx_batch import (
    build_flows_batched,
    build_flows_batched_v2,
    build_flows_batched_v3,
    step_batched,
)
from .mlx_genome import WEIGHTS_PER_GENOME
from .mlx_genome_v2 import WEIGHTS_PER_GENOME_V2
from .mlx_genome_v3 import WEIGHTS_PER_GENOME_V3
from .state import Flow, GameNode, GameState
from .vision import build_vision_table


def weights_per_genome(model_kind: str) -> int:
    if model_kind == "v1":
        return WEIGHTS_PER_GENOME
    if model_kind == "v2":
        return WEIGHTS_PER_GENOME_V2
    if model_kind == "v3":
        return WEIGHTS_PER_GENOME_V3
    raise ValueError(f"unknown model_kind: {model_kind!r}")

# A seat is either a genome (NN-driven) or a string name of a hand-coded AI.
Seat = Union[mx.array, str]

WIN_CHECK_EVERY_TICKS = 250


@dataclass
class GameConfig:
    radius: int = 18
    distance: int = 2
    num_players: int = 12
    ticks: int = 10000
    ai_period_ticks: int = 5
    early_weight: float = 0.5
    linger_penalty: float = 8.0
    model_kind: str = "v1"     # "v1" (91-feat, 2-hop) or "v2" (181-feat, 3-hop)


@dataclass
class RecordConfig:
    """Frame-recording knobs. tick_stride=0 disables recording."""
    tick_stride: int = 5
    dt_per_tick_ms: int = 100


@dataclass
class GameResult:
    fitness: np.ndarray            # (num_players,) float64
    frames: list[GameState]        # empty if not recorded
    final_cells: np.ndarray        # (num_players,) int — cell counts at decision tick
    decision_tick: int             # tick at which game became decided (or tick cap)


def _alive_player_count_per_game(owner_np: np.ndarray, P: int) -> np.ndarray:
    """(G,) int — number of players with ≥1 cell on each board."""
    G = owner_np.shape[0]
    counts = np.zeros(G, dtype=np.int64)
    for p in range(P):
        has_p = np.any(owner_np == p, axis=1)
        counts += has_p.astype(np.int64)
    return counts


def _to_python_state(
    base: GameState,
    owner_row: np.ndarray,
    strength_row: np.ndarray,
    flows: list[tuple[int, int, int]],
    tick: int,
) -> GameState:
    new_nodes = tuple(
        GameNode(
            id=n.id,
            pos=n.pos,
            owner=None if int(owner_row[i]) == -1 else int(owner_row[i]),
            strength=float(strength_row[i]),
        )
        for i, n in enumerate(base.nodes)
    )
    new_flows = tuple(Flow(src=s, dst=d, player=p) for (s, d, p) in flows)
    return GameState(
        nodes=new_nodes,
        edges=base.edges,
        flows=new_flows,
        tick=tick,
        num_players=base.num_players,
    )


def play_batch_games(
    rosters_list: list[list[Seat]],
    config: GameConfig,
    record_game_idx: Optional[int] = None,
    record: Optional[RecordConfig] = None,
) -> list[GameResult]:
    """Run G = len(rosters_list) games in parallel.

    `record_game_idx` (paired with `record`) names one game whose frames get
    recorded for replay output. Others run without recording (saves memory).
    """
    G = len(rosters_list)
    assert G > 0
    S = config.num_players
    for r in rosters_list:
        assert len(r) == S, "all rosters must have num_players seats"

    base = make_initial_state(config.radius, config.distance, config.num_players)
    neighbors_np = build_neighbor_table(base)
    N = len(base.nodes)
    P = config.num_players
    F_MAX = N  # at most one flow per source cell; bound by N
    do_record = record is not None and record_game_idx is not None
    model_kind = config.model_kind
    W = weights_per_genome(model_kind)
    vision_np = build_vision_table(config.radius) if model_kind == "v2" else None

    # Initial state, tiled across G.
    init_owner = np.array(
        [-1 if n.owner is None else n.owner for n in base.nodes], dtype=np.int32
    )
    init_strength = np.array([n.strength for n in base.nodes], dtype=np.float32)
    owner = mx.array(np.tile(init_owner, (G, 1)))
    strength = mx.array(np.tile(init_strength, (G, 1)))
    neighbors = mx.array(neighbors_np)
    live_mask = mx.array(np.ones(G, dtype=np.bool_))

    # Genome weights tensor (G, S, W). For hand-coded seats, plug a zero
    # placeholder — build_flows_batched overlays the aggressive policy on
    # cells owned by `aggressive_seat`, so the NN output for that seat is
    # discarded. We require all games to put aggressive at the same seat
    # index (or none of them); mixed configurations aren't supported.
    weights_np = np.zeros((G, S, W), dtype=np.float32)
    aggressive_seat = -1
    for g, roster in enumerate(rosters_list):
        for s, seat in enumerate(roster):
            if isinstance(seat, str):
                if seat != "aggressive":
                    raise ValueError(f"unknown hand-coded seat: {seat!r}")
                if aggressive_seat == -1:
                    aggressive_seat = s
                elif aggressive_seat != s:
                    raise ValueError(
                        f"aggressive seat must be consistent across batch "
                        f"(got {aggressive_seat} and {s})"
                    )
            else:
                arr = np.array(seat, copy=False)
                if arr.shape[0] != W:
                    raise ValueError(
                        f"genome size mismatch: got {arr.shape[0]}, "
                        f"expected {W} for model_kind={model_kind!r}"
                    )
                weights_np[g, s] = arr
    weights = mx.array(weights_np)
    vision = mx.array(vision_np) if vision_np is not None else None
    mx.eval(owner, strength, neighbors, live_mask, weights)
    if vision is not None:
        mx.eval(vision)

    decision_tick = np.full(G, config.ticks, dtype=np.int64)
    decision_cells: dict[int, np.ndarray] = {}     # g → (P,) at decision tick
    mid_cells: dict[int, np.ndarray] = {}          # g → (P,) at decision_tick // 2
    mid_captured = np.zeros(G, dtype=np.bool_)

    # Recorded frame buffer: stash MLX array refs during the loop, eval+convert
    # in one bulk pass at the end. Avoids per-frame mx.eval sync barriers.
    frames: list[GameState] = []
    pending_record: list[tuple[int, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]] = []

    if do_record:
        pending_record.append((
            0,
            owner[record_game_idx],
            strength[record_game_idx],
            mx.zeros((N,), dtype=mx.int32),
            mx.zeros((N,), dtype=mx.int32),
            mx.full((N,), -1, dtype=mx.int32),
            mx.zeros((N,), dtype=mx.bool_),
        ))

    def _capture_decision(g: int, tick: int, owner_np: np.ndarray) -> None:
        counts = np.zeros(P, dtype=np.int64)
        for p in range(P):
            counts[p] = int(np.sum(owner_np[g] == p))
        decision_cells[g] = counts
        decision_tick[g] = tick

    def _capture_mid_if_due(t: int, owner_np: np.ndarray) -> None:
        # For any live game whose mid-tick (decision_tick // 2 ≈ ticks // 2 since
        # not-yet-decided games default to ticks) hits now, capture cell counts.
        target_mid = config.ticks // 2
        if t != target_mid:
            return
        for g in range(G):
            if mid_captured[g]:
                continue
            counts = np.zeros(P, dtype=np.int64)
            for p in range(P):
                counts[p] = int(np.sum(owner_np[g] == p))
            mid_cells[g] = counts
            mid_captured[g] = True

    # Flow tensors are dense (G, N): one slot per source cell. They're rebuilt
    # entirely on each AI tick by build_flows_batched (no toggle persistence).
    flow_src = mx.broadcast_to(mx.arange(N).reshape(1, N), (G, N)).astype(mx.int32)
    flow_dst = mx.zeros((G, N), dtype=mx.int32)
    flow_player = mx.full((G, N), -1, dtype=mx.int32)
    flow_valid = mx.zeros((G, N), dtype=mx.bool_)

    for t in range(1, config.ticks + 1):
        if t % config.ai_period_ticks == 0:
            # Whole-GPU AI tick: NN forward + aggressive overlay + flow build.
            if model_kind == "v2":
                flow_src, flow_dst, flow_player, flow_valid = build_flows_batched_v2(
                    weights, owner, strength, neighbors, vision, aggressive_seat, P,
                )
            elif model_kind == "v3":
                flow_src, flow_dst, flow_player, flow_valid = build_flows_batched_v3(
                    weights, owner, strength, neighbors, aggressive_seat, P,
                )
            else:
                flow_src, flow_dst, flow_player, flow_valid = build_flows_batched(
                    weights, owner, strength, neighbors, aggressive_seat, P,
                )

        new_owner, new_strength, new_flow_valid = step_batched(
            owner, strength, flow_src, flow_dst, flow_player, flow_valid,
            live_mask, P, 0.1,
        )
        owner = new_owner
        strength = new_strength
        flow_valid = new_flow_valid

        # Periodic win check (every K ticks). Cheap reduction.
        if t % WIN_CHECK_EVERY_TICKS == 0 or t == config.ticks:
            mx.eval(owner, strength, flow_valid, live_mask)
            owner_np_now = np.array(owner, copy=False)
            alive = _alive_player_count_per_game(owner_np_now, P)
            lm_np = np.array(live_mask, copy=True)
            changed = False
            for g in range(G):
                if not lm_np[g]:
                    continue
                if alive[g] <= 1:
                    _capture_decision(g, t, owner_np_now)
                    lm_np[g] = False
                    changed = True
            if changed:
                live_mask = mx.array(lm_np)
            if not bool(lm_np.any()):
                _capture_mid_if_due(t, owner_np_now)
                break

        if t == config.ticks // 2:
            mx.eval(owner)
            _capture_mid_if_due(t, np.array(owner, copy=False))

        if do_record and t % record.tick_stride == 0:
            # Stash refs; eval+convert all at end of game.
            pending_record.append((
                t,
                owner[record_game_idx],
                strength[record_game_idx],
                flow_src[record_game_idx],
                flow_dst[record_game_idx],
                flow_player[record_game_idx],
                flow_valid[record_game_idx],
            ))

    # Fallthrough — for any game still live at tick cap, capture decision now.
    mx.eval(owner)
    owner_np_final = np.array(owner, copy=False)
    for g in range(G):
        if g not in decision_cells:
            counts = np.zeros(P, dtype=np.int64)
            for p in range(P):
                counts[p] = int(np.sum(owner_np_final[g] == p))
            decision_cells[g] = counts
            decision_tick[g] = config.ticks
        if not mid_captured[g]:
            # never reached mid_tick — game ended super fast; use decision-tick counts.
            mid_cells[g] = decision_cells[g].copy()

    # Bulk-eval recorded frames and build Python state list.
    if pending_record:
        flat: list[mx.array] = []
        for (_, o, s, fs, fd, fp, fv) in pending_record:
            flat.extend([o, s, fs, fd, fp, fv])
        mx.eval(*flat)
        for (tick, o, s, fs, fd, fp, fv) in pending_record:
            o_np = np.array(o, copy=False)
            s_np = np.array(s, copy=False)
            fv_np = np.array(fv, copy=False)
            if bool(fv_np.any()):
                fs_np = np.array(fs, copy=False)
                fd_np = np.array(fd, copy=False)
                fp_np = np.array(fp, copy=False)
                idxs = np.nonzero(fv_np)[0]
                rec_flows = [
                    (int(fs_np[i]), int(fd_np[i]), int(fp_np[i])) for i in idxs
                ]
            else:
                rec_flows = []
            frames.append(_to_python_state(base, o_np, s_np, rec_flows, tick))

    # Build results.
    results: list[GameResult] = []
    for g in range(G):
        final = decision_cells[g]
        mid = mid_cells[g]
        total_owned = int(final.sum())
        fit = np.zeros(P, dtype=np.float64)
        for p in range(P):
            opp = total_owned - int(final[p])
            fit[p] = (
                float(final[p])
                + config.early_weight * float(mid[p])
                - config.linger_penalty * float(opp)
            )
        results.append(GameResult(
            fitness=fit,
            frames=frames if (do_record and g == record_game_idx) else [],
            final_cells=final,
            decision_tick=int(decision_tick[g]),
        ))
    return results


def play_game(
    roster: list[Seat],
    config: GameConfig,
    record: Optional[RecordConfig] = None,
) -> GameResult:
    """Thin G=1 wrapper around play_batch_games."""
    record_idx = 0 if record is not None else None
    results = play_batch_games(
        [roster], config, record_game_idx=record_idx, record=record,
    )
    return results[0]
