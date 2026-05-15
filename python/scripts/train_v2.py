"""flux v2 PPO trainer.

Forked from train_ppo.py and adapted to the v2 PRD:

  - State carries persistent `edge_pressure` (G, N, K).
  - Action space is 13 (Set k=0..5 / Clear k=0..5 / no-op).
  - Reward is three terms: power Δ (per-seat strength delta), waste (per-seat
    attributed waste from the reducer), time (per-tick small constant penalty),
    plus a terminal win bonus.
  - Replay output is .flxr v2 with quantized edge_pressure on each flow.
  - Each rollout uses a random board: randomized seat placements + N dead
    hexes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx                                                # noqa: E402
import mlx.nn as nn                                                  # noqa: E402
import mlx.optimizers as optim                                       # noqa: E402
import numpy as np                                                   # noqa: E402

from flux_v2.graph import make_board, random_seat_and_dead            # noqa: E402
from flux_v2.mlx_step import apply_actions_batched, tick_batched      # noqa: E402
from flux_v2.ppo import AttnActorCritic, GNNActorCritic               # noqa: E402
from flux_v2.solver import solver_actions as bfs_solver_actions       # noqa: E402
from flux_v2.solver_lightning import lightning_solver_actions         # noqa: E402

MODEL_REGISTRY = {
    "gnn": GNNActorCritic,
    "attn": AttnActorCritic,
}

PRETRAIN_SOLVERS = {
    "bfs": bfs_solver_actions,
    "lightning": lambda s, seat, rng=None: lightning_solver_actions(s, seat, rng=rng, mode="max"),
    "lightning_sum": lambda s, seat, rng=None: lightning_solver_actions(s, seat, rng=rng, mode="sum"),
    "lightning_sum_pw": lambda s, seat, rng=None: lightning_solver_actions(s, seat, rng=rng, mode="sum_pw"),
    "lightning_loop": lambda s, seat, rng=None: lightning_solver_actions(s, seat, rng=rng, mode="loop"),
    "lightning_attn": lambda s, seat, rng=None: lightning_solver_actions(s, seat, rng=rng, mode="attn"),
}


def _ckpt_label(path) -> str:
    """Best-effort relative path for log messages — falls back to absolute
    when the checkpoint lives outside the repo (e.g., a job temp dir)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
from flux_v2.replay import (                                          # noqa: E402
    Frame,
    ReplayHeader,
    ReplayWriter,
    append_index,
)
from flux_v2.state import (                                           # noqa: E402
    ACTION_NOOP,
    DEAD,
    K,
    MAX_EDGE,
    MAX_STRENGTH,
    NEUTRAL,
    NUM_ACTIONS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "public" / "v2" / "replays"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "python" / "checkpoints" / "v2" / "latest.npz"


@dataclass
class PPOConfig:
    radius: int = 6
    num_players: int = 6
    games_per_rollout: int = 4
    ai_period_ticks: int = 5
    max_ticks: int = 10000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 512
    # Reward terms — composite power: territory held + damage dealt.
    # `power_held_coef * cells_owned` — per-tick reward for holding territory
    #   (non-saturating; captures bump it instantly, holding pays continuously)
    # `power_damage_coef * pressure_dealt_to_non_friendly` — per-tick reward
    #   for outflows pointing at enemies/neutrals (the "doing damage" signal
    #   that activates between captures)
    power_held_coef: float = 0.02
    power_damage_coef: float = 0.1
    # Legacy reward knobs (kept for backwards-compat experiments).
    power_coef: float = 0.0          # deprecated: per-unit Δ(Σ strength_owned)
    capture_coef: float = 0.0        # deprecated: per-cell Δ(cells_owned)
    waste_coef: float = 0.05         # per-unit attributed waste
    time_coef: float = 0.01          # per AI-tick small penalty
    win_bonus: float = 50.0          # terminal reward for the winner
    # Killer-instinct signal: per-tick reward proportional to the number of
    # already-eliminated seats. Solves the GAE-discounted terminal-win
    # problem (γ=0.99 over 2000 ticks crushes win_bonus to ~zero at tick 0).
    # Every dead enemy pays ongoing dividends; finishing weakened opponents
    # becomes economically motivated rather than a delayed gamble.
    kill_pressure_coef: float = 0.0  # 0 = off; ~5.0 makes one kill worth +5/tick
    num_dead_cells: int = 40
    # Convenience.
    record_stride: int = 25          # game ticks between recorded frames
    # Mixed-seat rollouts: opponent seats use a hand-written solver instead of
    # the model. Their seats still contribute to the environment (rewards,
    # captures, etc.) but their actions and gradient are masked out of the
    # PPO loss, so the model trains against a stationary opponent rather
    # than itself. Empty tuple = pure self-play.
    opponent_seats: tuple[int, ...] = ()
    opponent_solver_name: str = "lightning_sum"


def _categorical_sample(logits: mx.array, rng_key: mx.array) -> tuple[mx.array, mx.array]:
    rng_key, sub = mx.random.split(rng_key)
    gumbel = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape, key=sub) + 1e-9) + 1e-9)
    actions = (logits + gumbel).argmax(axis=-1).astype(mx.int32)
    return actions, rng_key


def _log_probs_at_actions(logits: mx.array, actions: mx.array) -> mx.array:
    log_softmax = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(log_softmax, actions[..., None], axis=-1).squeeze(-1)


def _alive_per_game(owner_np: np.ndarray, P: int) -> np.ndarray:
    counts = np.zeros(owner_np.shape[0], dtype=np.int64)
    for p in range(P):
        counts += np.any(owner_np == p, axis=1).astype(np.int64)
    return counts


def _per_seat_strength(owner_np: np.ndarray, strength_np: np.ndarray, P: int) -> np.ndarray:
    """Returns (G, P) total strength per seat."""
    G = owner_np.shape[0]
    out = np.zeros((G, P), dtype=np.float32)
    for p in range(P):
        out[:, p] = (strength_np * (owner_np == p)).sum(axis=1)
    return out


def _per_seat_damage_dealt(
    owner_np: np.ndarray,            # (G, N) int32
    outflow_np: np.ndarray,          # (G, N, K) bool
    edge_pressure_np: np.ndarray,    # (G, N, K) float32
    neighbors_np: np.ndarray,        # (N, K) int32
    P: int,
) -> np.ndarray:
    """Per-seat sum of edge_pressure on my-cell outflows targeting non-friendly
    cells. The "damage dealt" component of composite power — counts the work
    a seat is doing AGAINST enemies, even before captures materialize."""
    G, N = owner_np.shape
    K = outflow_np.shape[-1]
    nb_safe = np.maximum(neighbors_np, 0)
    nb_valid = (neighbors_np >= 0).reshape(1, N, K)
    owner_d = owner_np[:, nb_safe]                          # (G, N, K)
    out = np.zeros((G, P), dtype=np.float32)
    of_f = outflow_np.astype(np.float32)
    for s in range(P):
        is_mine = (owner_np == s).reshape(G, N, 1).astype(np.float32)
        is_non_friendly = ((owner_d != s) & nb_valid).astype(np.float32)
        contrib = edge_pressure_np * of_f * is_non_friendly * is_mine
        out[:, s] = contrib.sum(axis=(1, 2))
    return out


def _per_seat_waste(owner_np: np.ndarray, waste_np: np.ndarray, P: int) -> np.ndarray:
    G = owner_np.shape[0]
    out = np.zeros((G, P), dtype=np.float32)
    for p in range(P):
        out[:, p] = (waste_np * (owner_np == p)).sum(axis=1)
    return out


def _per_seat_cells(owner_np: np.ndarray, P: int) -> np.ndarray:
    G = owner_np.shape[0]
    out = np.zeros((G, P), dtype=np.int64)
    for p in range(P):
        out[:, p] = (owner_np == p).sum(axis=1)
    return out


# Player palette mirrors src/render/scene.ts so wandb images match the browser.
_PLAYER_PALETTE_RGB = np.array([
    [0x4a, 0x90, 0xe2], [0xe2, 0x4a, 0x4a], [0x4a, 0xe2, 0x8a], [0xe2, 0xc4, 0x4a],
    [0xa4, 0x4a, 0xe2], [0xe2, 0x88, 0x4a], [0x4a, 0xe2, 0xe2], [0xe2, 0x4a, 0x88],
    [0x88, 0xe2, 0x4a], [0xe2, 0xe2, 0x4a], [0x4a, 0x88, 0xe2], [0xa4, 0xe2, 0x4a],
], dtype=np.uint8)
_NEUTRAL_RGB = np.array([0x33, 0x33, 0x33], dtype=np.uint8)
_DEAD_RGB = np.array([0x66, 0x44, 0x55], dtype=np.uint8)


def _render_end_state(owner_end: np.ndarray, P: int) -> np.ndarray | None:
    G, N = owner_end.shape
    if G == 0 or N == 0:
        return None
    img = np.empty((G, N, 3), dtype=np.uint8)
    img[:] = _NEUTRAL_RGB
    for g in range(G):
        for s in range(P):
            mask = owner_end[g] == s
            img[g, mask] = _PLAYER_PALETTE_RGB[s % len(_PLAYER_PALETTE_RGB)]
        dead = owner_end[g] == DEAD
        img[g, dead] = _DEAD_RGB
    return img


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    owner: np.ndarray              # (T, G, N)
    strength: np.ndarray           # (T, G, N)
    outflow: np.ndarray            # (T, G, N, K)
    edge_pressure: np.ndarray      # (T, G, N, K) — pressure at decision time
    actions: np.ndarray            # (T, G, S, N)
    log_probs: np.ndarray          # (T, G, S)
    values: np.ndarray             # (T, G, S)
    rewards: np.ndarray            # (T, G, S)
    dones: np.ndarray              # (T, G)
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None
    # Per-rollout reward decomposition for wandb.
    reward_power: np.ndarray | None = None
    reward_capture: np.ndarray | None = None
    reward_waste: np.ndarray | None = None
    reward_time: np.ndarray | None = None
    reward_kill: np.ndarray | None = None


def make_batched_initial_state(
    config: PPOConfig, rng: np.random.Generator,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.ndarray, list[np.ndarray]]:
    """Build a (G, ...) initial state by sampling each game's board independently.

    Returns owner, strength, outflow, edge_pressure, alive_mask, neighbors,
    plus a list of per-game dead-cell index arrays for replay metadata.
    """
    G = config.games_per_rollout
    P = config.num_players
    base = make_board(config.radius, P)
    N = base.N
    neighbors_np = base.neighbors

    owner = np.full((G, N), NEUTRAL, dtype=np.int32)
    strength = np.zeros((G, N), dtype=np.float32)
    dead_lists: list[np.ndarray] = []
    for g in range(G):
        seats, dead = random_seat_and_dead(
            N, P, config.num_dead_cells, rng,
            neighbors=base.neighbors,
            min_seat_dist=2, coord=base.coord,
        )
        owner_g = np.full(N, NEUTRAL, dtype=np.int32)
        strength_g = np.full(N, 10.0, dtype=np.float32)
        if len(dead) > 0:
            owner_g[dead] = DEAD
            strength_g[dead] = 0.0
        for p, cell in enumerate(seats):
            owner_g[int(cell)] = p
            strength_g[int(cell)] = 30.0
        owner[g] = owner_g
        strength[g] = strength_g
        dead_lists.append(np.asarray(dead, dtype=np.int32))

    outflow = np.zeros((G, N, K), dtype=np.bool_)
    edge_pressure = np.zeros((G, N, K), dtype=np.float32)
    alive = np.ones(G, dtype=np.bool_)

    return (
        mx.array(owner), mx.array(strength), mx.array(outflow),
        mx.array(edge_pressure), mx.array(alive),
        mx.array(neighbors_np), dead_lists, base,
    )


def collect_rollout(
    model: GNNActorCritic,
    config: PPOConfig,
    rng_np: np.random.Generator,
    rng_mx: mx.array,
) -> tuple[Rollout, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
           mx.array, np.ndarray]:
    """Run G games to completion or max_ticks. Returns the rollout, a list of
    recorded frames for game 0 (each frame is (owner, strength, outflow,
    edge_pressure) numpy), the updated MLX rng key, and the dead-cell mask
    for game 0 (the recorded game)."""
    P = config.num_players

    owner, strength, outflow, edge_pressure, alive, neighbors, dead_lists, base_for_rollout = (
        make_batched_initial_state(config, rng_np)
    )
    G = config.games_per_rollout
    N = int(owner.shape[1])
    # NumPy view of the MLX neighbors tensor (static — same across all games).
    neighbors_np_local = np.array(neighbors, copy=False)

    # Storage.
    owners_log: list[np.ndarray] = []
    strengths_log: list[np.ndarray] = []
    outflows_log: list[np.ndarray] = []
    edge_pressures_log: list[np.ndarray] = []
    actions_log: list[np.ndarray] = []
    log_probs_log: list[np.ndarray] = []
    values_log: list[np.ndarray] = []
    rewards_log: list[np.ndarray] = []
    dones_log: list[np.ndarray] = []
    reward_power_log: list[np.ndarray] = []
    reward_capture_log: list[np.ndarray] = []
    reward_waste_log: list[np.ndarray] = []
    reward_time_log: list[np.ndarray] = []
    reward_kill_log: list[np.ndarray] = []

    frames_g0: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # Initial frame.
    mx.eval(owner, strength, outflow, edge_pressure)
    frames_g0.append((
        np.array(owner, copy=False)[0].copy(),
        np.array(strength, copy=False)[0].copy(),
        np.array(outflow, copy=False)[0].copy(),
        np.array(edge_pressure, copy=False)[0].copy(),
    ))

    waste_acc = mx.zeros((G, N), dtype=mx.float32)
    prev_strength_by_seat = _per_seat_strength(
        np.array(owner, copy=False), np.array(strength, copy=False), P,
    )
    prev_cells_by_seat = _per_seat_cells(np.array(owner, copy=False), P).astype(np.float32)
    # Per-killer kill-pressure tracking: cumulative attributed kills per
    # (game, seat). When seat p eliminates seat q, kills_per_seat[g, p] += 1
    # and stays incremented for the rest of the rollout — paying a per-tick
    # dividend `kill_pressure_coef * kills_per_seat[g, p]` to seat p.
    prev_owner_np = np.array(owner, copy=False).copy()
    prev_alive_per_seat = np.zeros((G, P), dtype=bool)
    for p in range(P):
        prev_alive_per_seat[:, p] = (prev_owner_np == p).any(axis=1)
    kills_per_seat = np.zeros((G, P), dtype=np.float32)

    for t in range(1, config.max_ticks + 1):
        # AI tick: sample actions, apply, log.
        if t % config.ai_period_ticks == 0:
            logits, value = model(
                owner, strength, outflow, edge_pressure, neighbors, P,
            )  # (G, S, N, A)
            actions, rng_mx = _categorical_sample(logits, rng_mx)         # (G, S, N)
            mx.eval(logits, value, actions, owner, strength, outflow, edge_pressure)

            owner_np = np.array(owner, copy=False)
            strength_np = np.array(strength, copy=False)
            outflow_np = np.array(outflow, copy=False)
            edge_pressure_np = np.array(edge_pressure, copy=False)

            # Each seat acts only on its own cells; we apply seat-by-seat in MLX.
            # Combine into a single per-cell action by selecting the action for
            # the cell's current owner. For unowned cells, action = NOOP.
            actions_np = np.array(actions, copy=False)

            # Mixed-seat rollouts: override the opponent seats' actions with
            # solver-chosen actions. The model still produced logits and
            # log_probs for those seats; we'll mask them out of the PPO loss.
            if config.opponent_seats:
                from flux_v2.state import State as _PyState
                solver_fn = PRETRAIN_SOLVERS[config.opponent_solver_name]
                for opp_s in config.opponent_seats:
                    for g in range(G):
                        sg = _PyState(
                            N=N, pos=base_for_rollout.pos,
                            coord=base_for_rollout.coord,
                            neighbors=neighbors_np_local,
                            owner=owner_np[g].copy(),
                            strength=strength_np[g].copy(),
                            outflow=outflow_np[g].copy(),
                            edge_pressure=edge_pressure_np[g].copy(),
                            tick=t, num_players=P,
                        )
                        actions_np[g, opp_s] = solver_fn(sg, opp_s, rng=rng_np)
                actions = mx.array(actions_np)
                mx.eval(actions)

            owner_safe = np.maximum(owner_np, 0)
            seat_idx = owner_safe[..., None]                              # (G, N, 1)
            combined = np.take_along_axis(
                actions_np.transpose(0, 2, 1),                            # (G, N, S)
                seat_idx, axis=-1,
            ).squeeze(-1)
            combined = np.where(owner_np >= 0, combined, ACTION_NOOP)

            new_outflow = apply_actions_batched(
                owner, outflow, mx.array(combined.astype(np.int32)), neighbors,
            )
            mx.eval(new_outflow)
            outflow = new_outflow

            # Log probability of the chosen action per (G, S, cell), summed over cells.
            log_probs_per_cell = _log_probs_at_actions(logits, actions)   # (G, S, N)
            log_probs_per_seat = log_probs_per_cell.sum(axis=-1)
            mx.eval(log_probs_per_seat)

            owners_log.append(owner_np.copy())
            strengths_log.append(strength_np.copy())
            outflows_log.append(outflow_np.copy())
            edge_pressures_log.append(edge_pressure_np.copy())
            actions_log.append(actions_np.copy())
            log_probs_log.append(np.array(log_probs_per_seat, copy=False).copy())
            values_log.append(np.array(value, copy=False).copy())
            live_np = np.array(alive, copy=False)
            dones_log.append((~live_np).copy())

        # Physics tick.
        owner, strength, outflow, edge_pressure, waste_per_cell = tick_batched(
            owner, strength, outflow, edge_pressure, alive, P, neighbors,
        )
        if t % config.ai_period_ticks == 1:
            waste_acc = waste_per_cell
        else:
            waste_acc = waste_acc + waste_per_cell

        # Frame recording for game 0.
        if t % config.record_stride == 0:
            mx.eval(owner, strength, outflow, edge_pressure)
            frames_g0.append((
                np.array(owner, copy=False)[0].copy(),
                np.array(strength, copy=False)[0].copy(),
                np.array(outflow, copy=False)[0].copy(),
                np.array(edge_pressure, copy=False)[0].copy(),
            ))

        # Reward at AI ticks (i.e. each apply_actions cadence step).
        if t % config.ai_period_ticks == 0:
            mx.eval(owner, strength, waste_acc)
            owner_np = np.array(owner, copy=False)
            strength_np = np.array(strength, copy=False)
            waste_np = np.array(waste_acc, copy=False)
            live_np = np.array(alive, copy=False)

            seat_strength_now = _per_seat_strength(owner_np, strength_np, P)
            seat_cells_now = _per_seat_cells(owner_np, P).astype(np.float32)
            seat_waste = _per_seat_waste(owner_np, waste_np, P)
            outflow_np_now = np.array(outflow, copy=False)
            edge_pressure_np_now = np.array(edge_pressure, copy=False)
            seat_damage = _per_seat_damage_dealt(
                owner_np, outflow_np_now, edge_pressure_np_now, neighbors_np_local, P,
            )
            delta_strength = seat_strength_now - prev_strength_by_seat
            # Capture reward: gains only (legacy; default 0 with composite power).
            delta_cells_gained = np.maximum(seat_cells_now - prev_cells_by_seat, 0.0)

            # Composite power: held territory + damage dealt to non-friendlies.
            # Both terms are per-tick (no saturation issue) and non-negative
            # for productive play. cells_owned rewards holding/expanding;
            # damage rewards actively pushing pressure into enemies/neutrals
            # even when no capture has materialized yet.
            r_power_held = config.power_held_coef * seat_cells_now
            r_power_damage = config.power_damage_coef * seat_damage
            r_power = (
                config.power_coef * delta_strength       # legacy, default 0
                + r_power_held
                + r_power_damage
            )
            r_capture = config.capture_coef * delta_cells_gained
            r_waste = -config.waste_coef * seat_waste
            r_time = -config.time_coef * np.ones_like(r_power)
            # Killer-instinct: attributed per-tick reward. When a seat
            # eliminates an enemy, that seat's `kills_per_seat` counter ticks
            # up by 1 and pays `kill_pressure_coef` per AI tick for the rest
            # of the rollout. Symmetric attribution by territory takeover:
            # if seat p ends up owning the most cells that were owned by the
            # dying seat at the previous tick, p is credited with the kill.
            alive_per_seat_now = np.zeros((G, P), dtype=bool)
            for p in range(P):
                alive_per_seat_now[:, p] = (owner_np == p).any(axis=1)
            newly_dead = prev_alive_per_seat & ~alive_per_seat_now      # (G, P)
            if newly_dead.any():
                for g in range(G):
                    if not live_np[g]:
                        continue
                    for q in range(P):
                        if not newly_dead[g, q]:
                            continue
                        # Cells previously owned by dying seat q.
                        was_q = (prev_owner_np[g] == q)
                        if not was_q.any():
                            continue
                        # Who owns them now? Find the seat with the most.
                        new_owners = owner_np[g, was_q]
                        # Restrict to live seats (ignore neutral/dead sentinels).
                        valid = new_owners >= 0
                        if not valid.any():
                            continue
                        new_owners_valid = new_owners[valid]
                        # Largest stake = killer.
                        counts = np.bincount(new_owners_valid, minlength=P)
                        killer = int(counts.argmax())
                        kills_per_seat[g, killer] += 1.0
            r_kill_pressure = config.kill_pressure_coef * kills_per_seat
            prev_owner_np = owner_np.copy()
            prev_alive_per_seat = alive_per_seat_now
            reward = r_power + r_capture + r_waste + r_time + r_kill_pressure
            # Zero rewards for already-dead games.
            mask = live_np.reshape(G, 1).astype(np.float32)
            reward = reward * mask
            r_power = r_power * mask
            r_capture = r_capture * mask
            r_waste = r_waste * mask
            r_time = r_time * mask
            r_kill_pressure = r_kill_pressure * mask
            rewards_log.append(reward.astype(np.float32))
            reward_power_log.append(r_power.astype(np.float32))
            reward_capture_log.append(r_capture.astype(np.float32))
            reward_waste_log.append(r_waste.astype(np.float32))
            reward_time_log.append(r_time.astype(np.float32))
            reward_kill_log.append(r_kill_pressure.astype(np.float32))
            prev_strength_by_seat = seat_strength_now
            prev_cells_by_seat = seat_cells_now

            # Game-over check.
            alive_counts = _alive_per_game(owner_np, P)
            lm = live_np.copy()
            for g in range(G):
                if not lm[g]:
                    continue
                if alive_counts[g] <= 1:
                    lm[g] = False
            if not lm.any():
                # Terminal win bonus.
                cells_end = _per_seat_cells(owner_np, P)
                winners = cells_end.argmax(axis=1)
                last_reward = rewards_log[-1]
                for g in range(G):
                    if cells_end[g, winners[g]] > 0:
                        last_reward[g, winners[g]] += config.win_bonus
                break
            if not np.array_equal(lm, live_np):
                alive = mx.array(lm)

    while len(rewards_log) < len(owners_log):
        rewards_log.append(np.zeros((G, P), dtype=np.float32))
        reward_power_log.append(np.zeros((G, P), dtype=np.float32))
        reward_capture_log.append(np.zeros((G, P), dtype=np.float32))
        reward_waste_log.append(np.zeros((G, P), dtype=np.float32))
        reward_time_log.append(np.zeros((G, P), dtype=np.float32))
        reward_kill_log.append(np.zeros((G, P), dtype=np.float32))

    T = len(owners_log)
    rollout = Rollout(
        owner=np.stack(owners_log, axis=0) if T else np.zeros((0, G, N), dtype=np.int32),
        strength=np.stack(strengths_log, axis=0) if T else np.zeros((0, G, N), dtype=np.float32),
        outflow=np.stack(outflows_log, axis=0) if T else np.zeros((0, G, N, K), dtype=np.bool_),
        edge_pressure=np.stack(edge_pressures_log, axis=0) if T else np.zeros((0, G, N, K), dtype=np.float32),
        actions=np.stack(actions_log, axis=0) if T else np.zeros((0, G, P, N), dtype=np.int32),
        log_probs=np.stack(log_probs_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
        values=np.stack(values_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
        rewards=np.stack(rewards_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
        dones=np.stack(dones_log, axis=0) if T else np.zeros((0, G), dtype=np.bool_),
        reward_power=np.stack(reward_power_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
        reward_capture=np.stack(reward_capture_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
        reward_waste=np.stack(reward_waste_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
        reward_time=np.stack(reward_time_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
        reward_kill=np.stack(reward_kill_log, axis=0) if T else np.zeros((0, G, P), dtype=np.float32),
    )
    rollout.advantages, rollout.returns = compute_gae(
        rollout, gamma=config.gamma, lam=config.gae_lambda,
    )

    return rollout, frames_g0, rng_mx, dead_lists[0]


def compute_gae(rollout: Rollout, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    T, G, S = rollout.values.shape
    adv = np.zeros_like(rollout.values)
    last_adv = np.zeros((G, S), dtype=np.float32)
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            next_v = np.zeros((G, S), dtype=np.float32)
            next_not_done = np.ones((G,), dtype=np.float32)
        else:
            next_v = rollout.values[t + 1]
            next_not_done = (~rollout.dones[t + 1]).astype(np.float32)
        delta = (
            rollout.rewards[t]
            + gamma * next_v * next_not_done.reshape(G, 1)
            - rollout.values[t]
        )
        last_adv = delta + gamma * lam * next_not_done.reshape(G, 1) * last_adv
        adv[t] = last_adv
    returns = adv + rollout.values
    adv_mean, adv_std = adv.mean(), adv.std() + 1e-8
    return ((adv - adv_mean) / adv_std).astype(np.float32), returns.astype(np.float32)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    model: GNNActorCritic,
    optimizer: optim.Optimizer,
    rollout: Rollout,
    neighbors: mx.array,
    config: PPOConfig,
) -> dict[str, float]:
    T, G, S, N = rollout.actions.shape
    P = config.num_players
    n_steps = T * G
    indices = np.arange(n_steps)

    flat_owner = rollout.owner.reshape(n_steps, N)
    flat_strength = rollout.strength.reshape(n_steps, N)
    flat_outflow = rollout.outflow.reshape(n_steps, N, K)
    flat_edge_pressure = rollout.edge_pressure.reshape(n_steps, N, K)
    flat_actions = rollout.actions.reshape(n_steps, S, N)
    flat_old_lp = rollout.log_probs.reshape(n_steps, S)
    flat_old_v = rollout.values.reshape(n_steps, S)
    flat_adv = rollout.advantages.reshape(n_steps, S)
    flat_ret = rollout.returns.reshape(n_steps, S)
    flat_dones = rollout.dones.reshape(n_steps)

    learn_mask = ~flat_dones

    # Mixed-seat: model_seat_mask[s] = 1.0 if seat s is model-controlled, else 0.0.
    # Used to mask the per-seat policy/value loss so opponent seats don't drive
    # gradient (they took solver actions, not model actions).
    if config.opponent_seats:
        opp = set(int(s) for s in config.opponent_seats)
        seat_mask_np = np.array(
            [0.0 if s in opp else 1.0 for s in range(S)], dtype=np.float32,
        )
    else:
        seat_mask_np = np.ones(S, dtype=np.float32)
    seat_mask_mx = mx.array(seat_mask_np)
    seat_mask_sum = float(seat_mask_np.sum())

    metrics: dict[str, float] = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
        "clip_fraction": 0.0, "ratio_mean": 0.0, "ratio_max": 0.0,
        "n_updates": 0,
    }

    metric_buf: list[tuple[mx.array, ...]] = []
    clip_eps = config.clip_eps

    def loss_fn(model, idx):
        owner_mb = mx.array(flat_owner[idx])
        strength_mb = mx.array(flat_strength[idx])
        outflow_mb = mx.array(flat_outflow[idx])
        edge_pressure_mb = mx.array(flat_edge_pressure[idx])
        actions_mb = mx.array(flat_actions[idx])
        old_lp_mb = mx.array(flat_old_lp[idx])
        adv_mb = mx.array(flat_adv[idx])
        ret_mb = mx.array(flat_ret[idx])

        logits, value = model(
            owner_mb, strength_mb, outflow_mb, edge_pressure_mb, neighbors, P,
        )
        log_softmax = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        new_lp_per_cell = mx.take_along_axis(
            log_softmax, actions_mb[..., None], axis=-1,
        ).squeeze(-1)
        new_lp = new_lp_per_cell.sum(axis=-1)

        ratio = mx.exp(new_lp - old_lp_mb)
        surr1 = ratio * adv_mb
        surr2 = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
        per_seat_pol = -mx.minimum(surr1, surr2)                        # (B, S)
        # Mask out opponent seats from policy and value loss.
        m = seat_mask_mx.reshape(1, S)
        denom = mx.maximum(m.sum() * per_seat_pol.shape[0], 1.0)
        policy_loss = (per_seat_pol * m).sum() / denom

        per_seat_val = (value - ret_mb) ** 2                            # (B, S)
        value_loss = (per_seat_val * m).sum() / denom

        probs = mx.softmax(logits, axis=-1)
        entropy_per_cell = -(probs * log_softmax).sum(axis=-1)
        entropy = entropy_per_cell.mean()

        approx_kl = (old_lp_mb - new_lp).mean()

        clipped = (mx.abs(ratio - 1.0) > clip_eps).astype(mx.float32)
        clip_fraction = clipped.mean()
        ratio_mean = ratio.mean()
        ratio_max = ratio.max()

        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        metric_buf.append((policy_loss, value_loss, entropy, approx_kl,
                           clip_fraction, ratio_mean, ratio_max))
        return loss

    grad_fn = nn.value_and_grad(model, loss_fn)
    last_grads: dict | None = None

    for epoch in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n_steps, config.minibatch_size):
            idx = indices[start:start + config.minibatch_size]
            idx = idx[learn_mask[idx]]
            if len(idx) == 0:
                continue
            loss, grads = grad_fn(model, idx)
            optimizer.update(model, grads)
            pl, vl, ent, kl, cf, rm, rmax = metric_buf.pop()
            mx.eval(model.parameters(), optimizer.state, pl, vl, ent, kl, cf, rm, rmax)
            metrics["policy_loss"] += float(pl)
            metrics["value_loss"] += float(vl)
            metrics["entropy"] += float(ent)
            metrics["approx_kl"] += float(kl)
            metrics["clip_fraction"] += float(cf)
            metrics["ratio_mean"] += float(rm)
            metrics["ratio_max"] = max(metrics["ratio_max"], float(rmax))
            metrics["n_updates"] += 1
            last_grads = grads

    if metrics["n_updates"] > 0:
        n = metrics["n_updates"]
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl",
                  "clip_fraction", "ratio_mean"):
            metrics[k] /= n

    def tree_l2(tree) -> float:
        acc = 0.0
        def walk(node):
            nonlocal acc
            if isinstance(node, dict):
                for v in node.values(): walk(v)
            elif isinstance(node, list):
                for v in node: walk(v)
            elif isinstance(node, mx.array):
                acc += float((node.astype(mx.float32) ** 2).sum())
        walk(tree)
        return float(np.sqrt(acc))

    metrics["weight_norm"] = tree_l2(model.parameters())
    metrics["grad_norm"] = tree_l2(last_grads) if last_grads is not None else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(model: GNNActorCritic, path: Path, generation: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    def walk(prefix: str, params):
        if isinstance(params, dict):
            for k, v in params.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(params, mx.array):
            flat[prefix] = np.array(params, copy=True)
        elif isinstance(params, list):
            for i, v in enumerate(params):
                walk(f"{prefix}.{i}", v)
    walk("", model.parameters())
    flat["__generation__"] = np.int64(generation)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **flat)
    os.replace(tmp, path)


def load_checkpoint(model: GNNActorCritic, path: Path) -> int:
    if not path.exists():
        return 0
    data = np.load(path, allow_pickle=False)
    params = model.parameters()
    def walk(prefix: str, container):
        if isinstance(container, dict):
            for k, v in container.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, mx.array):
                    if key in data.files:
                        container[k] = mx.array(data[key])
                elif isinstance(v, (dict, list)):
                    walk(key, v)
        elif isinstance(container, list):
            for i, v in enumerate(container):
                key = f"{prefix}.{i}"
                if isinstance(v, mx.array):
                    if key in data.files:
                        container[i] = mx.array(data[key])
                elif isinstance(v, (dict, list)):
                    walk(key, v)
    walk("", params)
    model.update(params)
    mx.eval(model.parameters())
    return int(data["__generation__"]) if "__generation__" in data.files else 0


# ---------------------------------------------------------------------------
# Replay write
# ---------------------------------------------------------------------------


def write_replay(
    path: Path,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    radius: int,
    num_players: int,
    num_nodes: int,
    record_stride: int,
    metadata: dict,
) -> None:
    if not frames:
        return
    header = ReplayHeader(
        radius=radius, num_players=num_players, num_nodes=num_nodes,
        tick_stride=record_stride, dt_per_tick_ms=100,
        metadata=metadata,
    )
    # Geometry is deterministic from (radius, num_players) — rebuild it once
    # so we can resolve outflow_slot → dst cell id when emitting flow records.
    from flux_v2.graph import make_board
    base = make_board(radius, num_players)
    neighbors_local = base.neighbors

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        w = ReplayWriter(f, header)
        for (owner_g, strength_g, outflow_g, ep_g) in frames:
            flows: list[tuple[int, int, int, float]] = []
            N_local = owner_g.shape[0]
            for c in range(N_local):
                if int(owner_g[c]) < 0:
                    continue
                for k in range(K):
                    if not bool(outflow_g[c, k]):
                        continue
                    d = int(neighbors_local[c, k])
                    if d < 0:
                        continue
                    flows.append((c, d, int(owner_g[c]), float(ep_g[c, k])))
            w.write_frame(_FrameLike(owner_g, strength_g, flows))
        w.close()
    os.replace(tmp, path)


class _FrameLike:
    """Shim so ReplayWriter.write_frame can take pre-built tuples."""
    def __init__(self, owner, strength, flows):
        self.owner = owner
        self.strength = strength
        self.flows = flows


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _structural_metrics(rollout: Rollout, neighbors_np: np.ndarray, P: int) -> dict[str, float]:
    """Territory-shape metrics that say what kind of game the policy is playing.

    All computed from end-of-rollout state (last logged AI tick):

    - frontier_ratio: per-seat fraction of owned cells adjacent to a
      non-friendly cell. Loops-only policy has this near zero; spread-out
      aggressors have it high.
    - interior_max_frac: per-seat fraction of non-frontier owned cells at
      MAX strength. "Back-line saturation" — should approach 1.0 in the
      dream scenario where interior loops pump max-strength reserves.
    - enemy_pressure_per_frontier: average pressure carried across
      non-friendly outflows on frontier cells. "Frontline pressure" signal.
    - expansion_rate: average cells gained per AI tick per seat (positive
      delta cells_owned, summed across rollout, divided by ticks).
    - neutral_capture_rate: average neutral cells consumed per AI tick per
      game (captured by any seat).
    """
    if rollout.owner.shape[0] == 0:
        return {
            "frontier_ratio_end": 0.0, "interior_max_frac_end": 0.0,
            "enemy_pressure_per_frontier_end": 0.0,
            "expansion_rate": 0.0, "neutral_capture_rate": 0.0,
            "total_edge_pressure_end": 0.0, "max_edge_pressure_end": 0.0,
        }

    owner_end = rollout.owner[-1]                                # (G, N)
    strength_end = rollout.strength[-1]                          # (G, N)
    outflow_end = rollout.outflow[-1]                            # (G, N, K)
    edge_pressure_end = rollout.edge_pressure[-1]                # (G, N, K)
    G, N = owner_end.shape

    # Per-(seat, cell) ownership.
    seat_idx = np.arange(P).reshape(1, P, 1)
    is_owned = (owner_end.reshape(G, 1, N) == seat_idx)          # (G, P, N)

    # Gather neighbor owners at end.
    nb_safe = np.maximum(neighbors_np, 0)                        # (N, K)
    nb_valid = (neighbors_np >= 0)                                # (N, K)
    owner_d = owner_end[:, nb_safe]                              # (G, N, K)

    # is_non_friendly[g, s, c, k] = neighbor valid AND owner_d != s
    nb_valid_b = nb_valid.reshape(1, 1, N, K)
    seat_idx_b = np.arange(P).reshape(1, P, 1, 1)
    owner_d_b = owner_d.reshape(G, 1, N, K)
    is_non_friendly = (owner_d_b != seat_idx_b) & nb_valid_b      # (G, P, N, K)
    has_non_friendly_nb = is_non_friendly.any(axis=-1)           # (G, P, N)

    # Frontier: owned and has a non-friendly neighbor.
    is_frontier = is_owned & has_non_friendly_nb                 # (G, P, N)
    is_interior = is_owned & np.logical_not(has_non_friendly_nb) # (G, P, N)

    owned_count = is_owned.sum(axis=-1).astype(np.float64)        # (G, P)
    frontier_count = is_frontier.sum(axis=-1).astype(np.float64)
    interior_count = is_interior.sum(axis=-1).astype(np.float64)

    frontier_ratio_per_seat = frontier_count / np.maximum(owned_count, 1.0)
    frontier_ratio = float(frontier_ratio_per_seat[owned_count > 0].mean()) \
        if (owned_count > 0).any() else 0.0

    # Interior-at-max.
    near_max = strength_end >= (MAX_STRENGTH * 0.99)
    near_max_b = near_max.reshape(G, 1, N)
    interior_max = (is_interior & near_max_b).sum(axis=-1).astype(np.float64)
    interior_max_frac_per_seat = interior_max / np.maximum(interior_count, 1.0)
    interior_max_frac = float(interior_max_frac_per_seat[interior_count > 0].mean()) \
        if (interior_count > 0).any() else 0.0

    # enemy_pressure: edge_pressure on outflows pointing at non-friendly neighbors.
    of_b = outflow_end.astype(np.float32).reshape(G, 1, N, K)
    ep_b = edge_pressure_end.reshape(G, 1, N, K)
    enemy_pressure_per_cell = (
        ep_b * of_b * is_non_friendly.astype(np.float32)
    ).sum(axis=-1)                                                # (G, P, N)
    enemy_pressure_frontier = (
        enemy_pressure_per_cell * is_frontier.astype(np.float32)
    ).sum(axis=-1)                                                # (G, P)
    enemy_pressure_per_frontier_per_seat = (
        enemy_pressure_frontier / np.maximum(frontier_count, 1.0)
    )
    enemy_pressure_per_frontier = float(
        enemy_pressure_per_frontier_per_seat[frontier_count > 0].mean()
    ) if (frontier_count > 0).any() else 0.0

    # Expansion + neutral-capture rates over the rollout.
    owner_arr = rollout.owner                                     # (T, G, N)
    T = owner_arr.shape[0]
    if T >= 2:
        # Per-seat cell counts over time.
        cells_over_time = np.zeros((T, G, P), dtype=np.int64)
        for s in range(P):
            cells_over_time[:, :, s] = (owner_arr == s).sum(axis=-1)
        cell_gains = np.maximum(np.diff(cells_over_time, axis=0), 0)  # (T-1, G, P)
        total_gains = cell_gains.sum(axis=0)                          # (G, P)
        expansion_rate = float(total_gains.mean() / max(T - 1, 1))

        neutrals_over_time = (owner_arr == -1).sum(axis=-1)           # (T, G)
        neutral_drops = np.maximum(-np.diff(neutrals_over_time, axis=0), 0)
        neutral_capture_rate = float(
            neutral_drops.sum(axis=0).mean() / max(T - 1, 1)
        )
    else:
        expansion_rate = 0.0
        neutral_capture_rate = 0.0

    # Total active edge pressure: sum of edge_pressure on active outflows,
    # averaged per game. In steady state with single-cell sending, this is
    # ~per-tick regen × num_active_edges. Combos (chained cells pushing
    # pressure through the same direction) cause edge_pressure to stack
    # higher than baseline. Climbing past the steady-state baseline is a
    # signal that the policy has found multi-cell pressure flows.
    active_edge_pressure = edge_pressure_end * outflow_end.astype(np.float32)
    total_edge_pressure = float(active_edge_pressure.sum() / max(G, 1))
    max_edge_pressure = float(active_edge_pressure.max()) if G > 0 else 0.0

    return {
        "frontier_ratio_end": frontier_ratio,
        "interior_max_frac_end": interior_max_frac,
        "enemy_pressure_per_frontier_end": enemy_pressure_per_frontier,
        "expansion_rate": expansion_rate,
        "neutral_capture_rate": neutral_capture_rate,
        "total_edge_pressure_end": total_edge_pressure,
        "max_edge_pressure_end": max_edge_pressure,
    }


def _rollout_diagnostics(rollout: Rollout, P: int) -> dict[str, float]:
    values = rollout.values
    returns = rollout.returns
    rewards = rollout.rewards
    actions = rollout.actions
    owner = rollout.owner
    dones = rollout.dones

    live = ~dones
    live_mask = np.broadcast_to(live[..., None], values.shape)
    if live_mask.any():
        v_live = values[live_mask]
        r_live = returns[live_mask]
        return_var = float(r_live.var())
        explained_var = (
            1.0 - float((r_live - v_live).var()) / max(return_var, 1e-8)
            if return_var > 0 else 0.0
        )
        value_mean = float(v_live.mean())
        value_std = float(v_live.std())
        return_mean = float(r_live.mean())
        return_std = float(r_live.std())
    else:
        explained_var = value_mean = value_std = return_mean = return_std = 0.0

    reward_flat = rewards[live].ravel() if live.any() else rewards.ravel()
    reward_step_mean = float(reward_flat.mean()) if reward_flat.size else 0.0
    reward_step_std = float(reward_flat.std()) if reward_flat.size else 0.0
    reward_step_max = float(reward_flat.max()) if reward_flat.size else 0.0
    reward_step_min = float(reward_flat.min()) if reward_flat.size else 0.0

    a = actions.ravel()
    if a.size:
        action_counts = np.bincount(a, minlength=NUM_ACTIONS).astype(np.float64)
        action_probs = action_counts / action_counts.sum()
        nz = action_probs > 0
        action_entropy = float(-(action_probs[nz] * np.log(action_probs[nz])).sum())
        action_pick_top_frac = float(action_probs.max())
        action_noop_frac = float(action_probs[ACTION_NOOP])
        # SET vs CLEAR fractions.
        set_frac = float(action_probs[:K].sum())
        clear_frac = float(action_probs[K:2 * K].sum())
    else:
        action_entropy = action_pick_top_frac = action_noop_frac = set_frac = clear_frac = 0.0

    end_owner = owner[-1]
    G, N = end_owner.shape
    cells = np.zeros((G, P), dtype=np.int64)
    for s in range(P):
        cells[:, s] = (end_owner == s).sum(axis=1)
    cells_max = cells.max(axis=1).astype(np.float64)
    cells_min = cells.min(axis=1).astype(np.float64)
    total_owned = cells.sum(axis=1).astype(np.float64)
    dominance = (cells_max / np.maximum(total_owned, 1.0))
    alive_seats = (cells > 0).sum(axis=1).astype(np.float64)
    neutral_frac = 1.0 - total_owned / N

    return {
        "explained_variance": explained_var,
        "value_mean": value_mean, "value_std": value_std,
        "return_mean": return_mean, "return_std": return_std,
        "reward_step_mean": reward_step_mean, "reward_step_std": reward_step_std,
        "reward_step_max": reward_step_max, "reward_step_min": reward_step_min,
        "action_entropy": action_entropy,
        "action_pick_top_frac": action_pick_top_frac,
        "action_noop_frac": action_noop_frac,
        "action_set_frac": set_frac,
        "action_clear_frac": clear_frac,
        "cells_max_end": float(cells_max.mean()),
        "cells_min_end": float(cells_min.mean()),
        "dominance": float(dominance.mean()),
        "alive_seats_end": float(alive_seats.mean()),
        "neutral_frac_end": float(neutral_frac.mean()),
    }


# ---------------------------------------------------------------------------
# Warmstart: supervised pretrain from a solver
# ---------------------------------------------------------------------------


def collect_solver_pretrain_data(
    config: PPOConfig,
    solver_fn,
    rng_np: np.random.Generator,
    base_board,
) -> tuple[list, list]:
    """Run G games with `solver_fn` driving all seats. Records (state, action)
    pairs at each AI tick. Used as supervised data for warmstart.

    Returns
    -------
    states_log : list of dict with keys owner/strength/outflow/edge_pressure
                 (all MLX tensors of shape (G, ...))
    actions_log : list of MLX (G, S, N) int32 — solver-chosen actions per cell
                  per seat. Non-owned cells get ACTION_NOOP.
    """
    from flux_v2.state import State as PyState
    P = config.num_players
    owner, strength, outflow, edge_pressure, alive, neighbors, dead_lists, _ = (
        make_batched_initial_state(config, rng_np)
    )
    G = config.games_per_rollout
    N = int(owner.shape[1])
    neighbors_np = np.array(neighbors, copy=False)
    pos_np = base_board.pos
    coord_np = base_board.coord

    states_log: list[dict] = []
    actions_log: list[mx.array] = []

    for t in range(1, config.max_ticks + 1):
        if t % config.ai_period_ticks == 0:
            mx.eval(owner, strength, outflow, edge_pressure)
            owner_np = np.array(owner, copy=False)
            strength_np = np.array(strength, copy=False)
            outflow_np = np.array(outflow, copy=False)
            ep_np = np.array(edge_pressure, copy=False)

            # Build per-(g, s) action via solver. Solver returns (N,) int32.
            actions_np = np.full((G, P, N), ACTION_NOOP, dtype=np.int32)
            for g in range(G):
                sg = PyState(
                    N=N, pos=pos_np, coord=coord_np, neighbors=neighbors_np,
                    owner=owner_np[g].copy(),
                    strength=strength_np[g].copy(),
                    outflow=outflow_np[g].copy(),
                    edge_pressure=ep_np[g].copy(),
                    tick=t, num_players=P,
                )
                for s in range(P):
                    a = solver_fn(sg, s, rng=rng_np)
                    actions_np[g, s] = a

            states_log.append({
                "owner": owner, "strength": strength,
                "outflow": outflow, "edge_pressure": edge_pressure,
            })
            actions_log.append(mx.array(actions_np))

            # Combine into per-cell action: each cell uses its owner's chosen action.
            owner_safe = np.maximum(owner_np, 0)
            seat_idx = owner_safe[..., None]
            combined = np.take_along_axis(
                actions_np.transpose(0, 2, 1), seat_idx, axis=-1,
            ).squeeze(-1)
            combined = np.where(owner_np >= 0, combined, ACTION_NOOP)

            new_outflow = apply_actions_batched(
                owner, outflow, mx.array(combined.astype(np.int32)), neighbors,
            )
            mx.eval(new_outflow)
            outflow = new_outflow

        # Physics tick.
        owner, strength, outflow, edge_pressure, _waste = tick_batched(
            owner, strength, outflow, edge_pressure, alive, P, neighbors,
        )

        # Optional: early-stop on alive==False for any seat — keeps data clean.
        # (we don't worry about this for warmstart data quality.)

    return states_log, actions_log


def pretrain_supervised(
    model,
    optimizer: optim.Adam,
    states_log: list,
    actions_log: list,
    neighbors_mx: mx.array,
    num_players: int,
    epochs: int,
) -> None:
    """Cross-entropy fit: model predicts solver's action choice per cell.
    Only owned cells contribute to the loss (solver returns NOOP elsewhere
    and we don't want to over-fit that)."""

    def cell_loss(model, state, actions):
        logits, _ = model(
            state["owner"], state["strength"], state["outflow"],
            state["edge_pressure"], neighbors_mx, num_players,
        )  # (G, S, N, A)
        log_probs = _log_probs_at_actions(logits, actions)              # (G, S, N)
        # Mask: only cells owned by the seat contribute.
        G = state["owner"].shape[0]
        S = num_players
        seat_idx = mx.arange(S).reshape(1, S, 1)
        is_owned = (state["owner"].reshape(G, 1, -1) == seat_idx)
        mask = is_owned.astype(mx.float32)
        denom = mx.maximum(mask.sum(), 1.0)
        loss = -(log_probs * mask).sum() / denom
        return loss

    grad_fn = nn.value_and_grad(model, cell_loss)
    for ep in range(epochs):
        ep_loss = 0.0
        n_steps = 0
        for state, actions in zip(states_log, actions_log):
            loss, grads = grad_fn(model, state, actions)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            ep_loss += float(loss)
            n_steps += 1
        print(f"  pretrain epoch {ep + 1}: mean_loss={ep_loss / max(n_steps, 1):.4f} "
              f"(over {n_steps} AI-tick batches)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--games-per-rollout", type=int, default=4)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=10000)
    ap.add_argument("--record-stride", type=int, default=25)
    ap.add_argument("--num-dead-cells", type=int, default=40)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=512)
    ap.add_argument("--power-coef", type=float, default=0.0,
                    help="(legacy) Δ(Σ strength_owned). Default 0 — composite "
                         "power replaces this.")
    ap.add_argument("--capture-coef", type=float, default=0.0,
                    help="(legacy) reward per net cell gained. Default 0 — "
                         "subsumed by power_held_coef in composite power.")
    ap.add_argument("--power-held-coef", type=float, default=0.02,
                    help="composite power: reward per cell_owned per AI tick. "
                         "Non-saturating territory signal.")
    ap.add_argument("--power-damage-coef", type=float, default=0.1,
                    help="composite power: reward per unit of edge_pressure on "
                         "outflows pointing at non-friendly cells. Rewards "
                         "active attack work even before captures land.")
    ap.add_argument("--waste-coef", type=float, default=0.05)
    ap.add_argument("--time-coef", type=float, default=0.01)
    ap.add_argument("--win-bonus", type=float, default=50.0)
    ap.add_argument("--kill-pressure-coef", type=float, default=0.0,
                    help="per-tick reward per dead enemy seat (shared across "
                         "surviving seats). Fixes GAE-killed terminal win bonus "
                         "by making 'finish weakened enemies' pay ongoing.")
    ap.add_argument("--iterations", type=int, default=0,
                    help="stop after N iterations (0 = forever)")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="flux-v2")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--model", choices=tuple(MODEL_REGISTRY), default="gnn",
                    help="actor-critic architecture: 'gnn' (default 3-layer GCN, "
                         "13-action softmax) or 'attn' (same backbone, SET-action "
                         "logits = (1-α)·attack_q + α·loop_q from learned heads)")
    ap.add_argument("--pretrain-epochs", type=int, default=0,
                    help="Run supervised behavior-cloning warmstart from a solver "
                         "BEFORE the PPO loop. 0 = no warmstart.")
    ap.add_argument("--pretrain-solver", choices=tuple(PRETRAIN_SOLVERS),
                    default="lightning_sum",
                    help="Solver to clone during warmstart.")
    ap.add_argument("--pretrain-games", type=int, default=0,
                    help="Number of games_per_rollout for the warmstart data "
                         "collection pass. 0 = use --games-per-rollout.")
    ap.add_argument("--opponent-seats", type=str, default="",
                    help="Comma-separated seat IDs (e.g. '0,2,4') that act via "
                         "a fixed solver instead of the model. The PPO loss "
                         "masks these seats — model trains against a "
                         "stationary opponent rather than itself.")
    ap.add_argument("--opponent-solver", choices=tuple(PRETRAIN_SOLVERS),
                    default="lightning_sum",
                    help="Solver controlling the opponent seats.")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINT_PATH

    config = PPOConfig(
        radius=args.radius, num_players=args.num_players,
        games_per_rollout=args.games_per_rollout,
        ai_period_ticks=args.ai_period_ticks, max_ticks=args.max_ticks,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps, entropy_coef=args.entropy_coef,
        value_coef=args.value_coef, lr=args.lr,
        update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
        power_coef=args.power_coef, capture_coef=args.capture_coef,
        power_held_coef=args.power_held_coef,
        power_damage_coef=args.power_damage_coef,
        waste_coef=args.waste_coef,
        time_coef=args.time_coef, win_bonus=args.win_bonus,
        kill_pressure_coef=args.kill_pressure_coef,
        num_dead_cells=args.num_dead_cells,
        record_stride=args.record_stride,
        opponent_seats=tuple(int(s) for s in args.opponent_seats.split(",") if s.strip()),
        opponent_solver_name=args.opponent_solver,
    )
    if config.opponent_seats:
        if not set(config.opponent_seats).issubset(range(config.num_players)):
            raise SystemExit(
                f"--opponent-seats {config.opponent_seats} out of range for "
                f"P={config.num_players}"
            )
        print(f"  mixed-seat: opponent seats {list(config.opponent_seats)} "
              f"use {config.opponent_solver_name}; "
              f"model trains on {config.num_players - len(config.opponent_seats)} seats")

    base = make_board(args.radius, args.num_players)
    N = base.N
    neighbors_mx = mx.array(base.neighbors)

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls()
    print(f"  model: {args.model} ({model_cls.__name__})")
    optimizer = optim.Adam(learning_rate=args.lr)
    mx.eval(model.parameters())

    iteration_offset = 0
    if not args.fresh:
        iteration_offset = load_checkpoint(model, args.checkpoint)
        if iteration_offset > 0:
            print(f"resumed from {_ckpt_label(args.checkpoint)} at iter {iteration_offset}")
    if iteration_offset == 0:
        print("starting from fresh policy")

    rng_mx = mx.random.key(args.seed)
    rng_np = np.random.default_rng(args.seed ^ 0xDEADBEEF)

    # Warmstart: behavior-clone a solver before the PPO loop.
    if args.pretrain_epochs > 0 and iteration_offset == 0:
        solver_fn = PRETRAIN_SOLVERS[args.pretrain_solver]
        pretrain_g = args.pretrain_games or args.games_per_rollout
        pretrain_cfg = PPOConfig(
            radius=config.radius, num_players=config.num_players,
            games_per_rollout=pretrain_g,
            ai_period_ticks=config.ai_period_ticks,
            max_ticks=config.max_ticks,
            num_dead_cells=config.num_dead_cells,
            record_stride=config.record_stride,
            gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_eps=config.clip_eps, entropy_coef=config.entropy_coef,
            value_coef=config.value_coef, lr=config.lr,
            update_epochs=config.update_epochs,
            minibatch_size=config.minibatch_size,
            power_coef=config.power_coef, capture_coef=config.capture_coef,
            power_held_coef=config.power_held_coef,
            power_damage_coef=config.power_damage_coef,
            waste_coef=config.waste_coef,
            time_coef=config.time_coef, win_bonus=config.win_bonus,
            kill_pressure_coef=config.kill_pressure_coef,
        )
        print(f"  warmstart: cloning {args.pretrain_solver} for {args.pretrain_epochs} "
              f"epoch(s) on {pretrain_g} games × {config.max_ticks} ticks")
        t_warm_collect = time.time()
        states_log, actions_log = collect_solver_pretrain_data(
            pretrain_cfg, solver_fn, rng_np, base,
        )
        print(f"  collected {len(states_log)} AI-tick batches in "
              f"{time.time() - t_warm_collect:.1f}s — supervised loss next")
        t_warm_train = time.time()
        pretrain_supervised(
            model, optimizer, states_log, actions_log, neighbors_mx,
            config.num_players, args.pretrain_epochs,
        )
        print(f"  warmstart done in {time.time() - t_warm_train:.1f}s")

    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or f"v2-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print(f"  wandb run: {wandb_run.url}")

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v2-io")

    print(f"training v2 PPO: radius={args.radius} N={N} G={args.games_per_rollout} "
          f"seats={args.num_players} max_ticks={args.max_ticks} dead={args.num_dead_cells}")

    iteration = iteration_offset
    try:
        while True:
            if args.iterations > 0 and (iteration - iteration_offset) >= args.iterations:
                break
            iteration += 1
            t0 = time.time()
            rollout, frames_g0, rng_mx, dead_g0 = collect_rollout(
                model, config, rng_np, rng_mx,
            )
            t_rollout = time.time() - t0

            t1 = time.time()
            metrics = ppo_update(model, optimizer, rollout, neighbors_mx, config)
            t_update = time.time() - t1

            mean_reward = float(rollout.rewards.sum(axis=0).mean())
            mean_power = float(rollout.reward_power.sum(axis=0).mean())
            mean_capture = float(rollout.reward_capture.sum(axis=0).mean())
            mean_waste = float(rollout.reward_waste.sum(axis=0).mean())
            mean_time = float(rollout.reward_time.sum(axis=0).mean())
            mean_kill = float(rollout.reward_kill.sum(axis=0).mean())
            total_steps = rollout.actions.shape[0]
            roll_diag = _rollout_diagnostics(rollout, config.num_players)
            struct = _structural_metrics(rollout, base.neighbors, config.num_players)
            print(
                f"iter {iteration:4d}  rollout={t_rollout:.2f}s update={t_update:.2f}s "
                f"steps={total_steps}  R={mean_reward:.2f} "
                f"(pwr={mean_power:.1f} cap={mean_capture:.1f} kill={mean_kill:.1f} wst={mean_waste:.1f} t={mean_time:.1f})  "
                f"pol={metrics['policy_loss']:.4f} ent={metrics['entropy']:.3f} "
                f"kl={metrics['approx_kl']:.4f} ev={roll_diag['explained_variance']:.2f}"
            )

            if wandb_run is not None:
                payload = {
                    "iteration": iteration,
                    "rollout_sec": t_rollout, "update_sec": t_update,
                    "rollout_steps": total_steps,
                    "mean_total_reward": mean_reward,
                    "reward_power_iter": mean_power,
                    "reward_capture_iter": mean_capture,
                    "reward_waste_iter": mean_waste,
                    "reward_time_iter": mean_time,
                    "reward_kill_iter": mean_kill,
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "approx_kl": metrics["approx_kl"],
                    "clip_fraction": metrics["clip_fraction"],
                    "ratio_mean": metrics["ratio_mean"],
                    "ratio_max": metrics["ratio_max"],
                    "grad_norm": metrics["grad_norm"],
                    "weight_norm": metrics["weight_norm"],
                    "explained_variance": roll_diag["explained_variance"],
                    "value_mean": roll_diag["value_mean"],
                    "value_std": roll_diag["value_std"],
                    "return_mean": roll_diag["return_mean"],
                    "return_std": roll_diag["return_std"],
                    "reward_step_mean": roll_diag["reward_step_mean"],
                    "reward_step_std": roll_diag["reward_step_std"],
                    "reward_step_max": roll_diag["reward_step_max"],
                    "reward_step_min": roll_diag["reward_step_min"],
                    "action_entropy": roll_diag["action_entropy"],
                    "action_pick_top_frac": roll_diag["action_pick_top_frac"],
                    "action_noop_frac": roll_diag["action_noop_frac"],
                    "action_set_frac": roll_diag["action_set_frac"],
                    "action_clear_frac": roll_diag["action_clear_frac"],
                    "cells_max_end": roll_diag["cells_max_end"],
                    "cells_min_end": roll_diag["cells_min_end"],
                    "dominance": roll_diag["dominance"],
                    "alive_seats_end": roll_diag["alive_seats_end"],
                    "neutral_frac_end": roll_diag["neutral_frac_end"],
                    # --- structural metrics (the user's "dream scenario" KPIs) ---
                    "frontier_ratio_end": struct["frontier_ratio_end"],
                    "interior_max_frac_end": struct["interior_max_frac_end"],
                    "enemy_pressure_per_frontier_end": struct["enemy_pressure_per_frontier_end"],
                    "expansion_rate": struct["expansion_rate"],
                    "neutral_capture_rate": struct["neutral_capture_rate"],
                    "total_edge_pressure_end": struct["total_edge_pressure_end"],
                    "max_edge_pressure_end": struct["max_edge_pressure_end"],
                }
                if iteration % 20 == 0 and rollout.owner.shape[0] > 0:
                    img = _render_end_state(rollout.owner[-1], config.num_players)
                    if img is not None:
                        try:
                            payload["end_state"] = wandb.Image(
                                img, caption=f"iter {iteration} end"
                            )
                        except Exception as e:
                            print(f"  (skipping end_state image: {e})")
                wandb_run.log(payload, step=iteration)

            # Replay write.
            if frames_g0:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                name = f"train_v2_{stamp}_i{iteration:05d}.flxr"
                replay_path = out_dir / name
                dead_g0_list = [int(x) for x in dead_g0]
                metadata = {
                    "kind": "train_v2",
                    "model": "v2",
                    "ruleset": "v2-pressure",
                    "iteration": iteration,
                    "generation": iteration,
                    "best_fitness": mean_reward,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "dead_cells": dead_g0_list,
                    "power_reward": mean_power,
                    "waste_reward": mean_waste,
                    "time_reward": mean_time,
                }
                frames_snapshot = frames_g0

                def _do_write(
                    p, fr, r, np_, ns, ts, m,
                    _it=iteration,
                ):
                    try:
                        write_replay(p, fr, r, np_, ns, ts, m)
                        append_index(out_dir, {
                            "file": p.name, "saved_at": m["saved_at"],
                            "kind": "train_v2", "model": "v2",
                            "ruleset": "v2-pressure",
                            "iteration": _it, "generation": _it,
                            "radius": r, "num_players": np_,
                        })
                    except Exception as e:
                        import traceback
                        print(f"  REPLAY WRITE FAILED iter {_it}: {type(e).__name__}: {e}")
                        traceback.print_exc()
                write_executor.submit(
                    _do_write, replay_path, frames_snapshot,
                    args.radius, args.num_players, N, args.record_stride,
                    metadata,
                )

            if iteration % 5 == 0:
                write_executor.submit(save_checkpoint, model, args.checkpoint, iteration)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        write_executor.shutdown(wait=True)
        save_checkpoint(model, args.checkpoint, iteration)
        print(f"checkpoint saved → {_ckpt_label(args.checkpoint)}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
