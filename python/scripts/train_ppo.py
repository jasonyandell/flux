"""PPO + GNN training loop for flux.

Rollout strategy: G parallel games × per-AI-tick decisions × 12 seats = a lot
of trajectories per iteration. All seats share one policy; the per-(seat, cell)
input features are seat-relative so the same parameters work everywhere.

Reward shaping (the load-bearing experiment): per-AI-tick cell delta per seat
gives a dense gradient signal. Final-cells terminal bonus optional.

Aggressive seat (when --aggressive-seat ≥ 0) plays a hand-coded policy and is
excluded from the PPO update — it's a fixed opponent, not a learner.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402

from flux.genome import build_neighbor_table  # noqa: E402
from flux.graph import make_initial_state  # noqa: E402
from flux.mlx_batch import (  # noqa: E402
    build_flows_from_actions,
    step_batched,
)
from flux.mlx_step_regen import (  # noqa: E402
    MAX_OUTPUT_PER_SEC,
    REGEN_BASE_PER_SEC,
    REGEN_SLOPE,
    step_batched_regen,
)
from flux.lookahead import lookahead_action_select  # noqa: E402
from flux.ppo import GNNActorCritic  # noqa: E402
from flux.replay import ReplayHeader, ReplayWriter  # noqa: E402
from flux.state import Flow, GameNode, GameState, MAX_STRENGTH  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "public" / "replays"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "python" / "checkpoints" / "ppo" / "latest.npz"


@dataclass
class PPOConfig:
    radius: int = 9
    distance: int = 2
    num_players: int = 12
    games_per_rollout: int = 4
    ai_period_ticks: int = 5
    max_ticks: int = 5000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 128
    aggressive_seat: int = 0
    cell_delta_reward: bool = True
    terminal_win_bonus: float = 50.0
    # Dense per-tick shaping. All terms range 0..1 per (game, seat) per AI tick.
    engagement_coef: float = 0.01    # reward: cells_sending / cells_owned
    idle_capped_coef: float = 0.02   # penalty: idle cells near cap / cells_owned
    idle_cap_threshold: float = 90.0
    output_boost_coef: float = 0.05  # reward: avg output_rate / MAX_OUTPUT_PER_SEC (regen-flow only)
    wasted_flux_coef: float = 0.0    # penalty per friendly-inflow unit clipped at MAX (regen-flow only)


def _categorical_sample(logits: mx.array, rng_key: mx.array) -> tuple[mx.array, mx.array]:
    """Sample categorical actions from logits using gumbel-max trick.
    Returns (actions, rng_key_new). actions has shape logits.shape[:-1] int32."""
    rng_key, sub = mx.random.split(rng_key)
    gumbel = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape, key=sub) + 1e-9) + 1e-9)
    actions = (logits + gumbel).argmax(axis=-1).astype(mx.int32)
    return actions, rng_key


def _log_probs_at_actions(logits: mx.array, actions: mx.array) -> mx.array:
    """Returns log p(action) for each (..., cell). logits: (..., A), actions: (...)."""
    log_softmax = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(log_softmax, actions[..., None], axis=-1).squeeze(-1)


def _alive_per_game(owner_np: np.ndarray, P: int) -> np.ndarray:
    counts = np.zeros(owner_np.shape[0], dtype=np.int64)
    for p in range(P):
        counts += np.any(owner_np == p, axis=1).astype(np.int64)
    return counts


def _cells_per_seat(owner_np: np.ndarray, P: int) -> np.ndarray:
    """Returns (G, P) int."""
    G = owner_np.shape[0]
    out = np.zeros((G, P), dtype=np.int64)
    for p in range(P):
        out[:, p] = (owner_np == p).sum(axis=1)
    return out


def _aggressive_actions_np(
    owner_np: np.ndarray,
    strength_np: np.ndarray,
    neighbors_np: np.ndarray,
    aggressive_seat: int,
    STRIDE: int = 18,
    threshold: float = 5.0,
) -> np.ndarray:
    """Vectorized per-game per-cell action override for the aggressive seat.
    Returns (G, N) int32: action for cells owned by aggressive_seat is the
    argmin-over-non-friendly-neighbor-strengths; for other cells, 18 (noop).

    Maxed cells fan out automatically at the simulation level (regardless of
    the action picked here), so we don't need spill-when-surrounded logic.
    """
    G, N = owner_np.shape
    nb_ids = neighbors_np.reshape(N, STRIDE).astype(np.int64)
    nb_valid = nb_ids >= 0                                              # (N, STRIDE)
    safe = np.maximum(nb_ids, 0)                                        # (N, STRIDE)

    nb_strength = strength_np[:, safe]                                  # (G, N, STRIDE)
    nb_owner = owner_np[:, safe]                                        # (G, N, STRIDE)

    is_non_friend = (nb_owner != aggressive_seat) & nb_valid.reshape(1, N, STRIDE)
    BIG = 1.0e9
    masked = np.where(is_non_friend, nb_strength, BIG)                  # (G, N, STRIDE)
    best_action = masked.argmin(axis=-1).astype(np.int32)               # (G, N)
    has_target = masked.min(axis=-1) < BIG / 2.0

    own_active = (
        (owner_np == aggressive_seat)
        & (strength_np > threshold)
        & has_target
    )
    return np.where(own_active, best_action, np.int32(18))


@dataclass
class Rollout:
    # All tensors stored as numpy (CPU) for the update loop's minibatch slicing.
    # Shapes: T = number of AI ticks captured.
    owner: np.ndarray         # (T, G, N) int32
    strength: np.ndarray      # (T, G, N) float32
    actions: np.ndarray       # (T, G, S, N) int32
    log_probs: np.ndarray     # (T, G, S) float32 — summed over cells per (game, seat)
    values: np.ndarray        # (T, G, S) float32
    rewards: np.ndarray       # (T, G, S) float32
    dones: np.ndarray         # (T, G) bool — whether the game was already over before this AI tick
    advantages: np.ndarray | None = None  # (T, G, S) float32, computed later
    returns: np.ndarray | None = None     # (T, G, S) float32
    dead_mask: np.ndarray | None = None   # (G, N) bool — fixed dead cells per game (None = no dead)


def collect_rollout(
    model: GNNActorCritic,
    config: PPOConfig,
    neighbors_np: np.ndarray,
    rng_key: mx.array,
    step_fn=step_batched,
    record_stride: int = 10,
    num_dead_cells: int = 0,
    randomize_starts: bool = False,
    lookahead_k: int = 0,
) -> tuple[Rollout, list[GameState], mx.array, np.ndarray]:
    """Run G games in parallel for up to config.max_ticks ticks. Returns the
    rollout buffer + a list of frames (for the recorded game = index 0) + the
    updated rng key + per-game dead-cell masks (G, N) bool."""
    G = config.games_per_rollout
    N = neighbors_np.shape[0] // 18
    P = config.num_players
    S = P
    aggressive_seat = config.aggressive_seat

    base = make_initial_state(config.radius, config.distance, P)
    default_init_strength = np.array(
        [n.strength for n in base.nodes], dtype=np.float32
    )
    # Per-game initial state: random seat placements + random dead cells.
    # Falls back to the deterministic make_initial_state layout when neither
    # randomization is enabled.
    init_owner_np = np.full((G, N), -1, dtype=np.int32)
    init_strength_np = np.tile(default_init_strength, (G, 1)).astype(np.float32)
    dead_mask_np = np.zeros((G, N), dtype=np.bool_)
    if num_dead_cells > 0 or randomize_starts:
        seat_base_strength = 30.0  # match make_initial_state's perimeter seat init
        for g in range(G):
            # Choose dead cells first; they're excluded from seat placement.
            avail = np.arange(N)
            if num_dead_cells > 0:
                dead = np.random.choice(N, size=num_dead_cells, replace=False)
                dead_mask_np[g, dead] = True
                init_strength_np[g, dead] = 0.0
                avail = np.setdiff1d(avail, dead, assume_unique=True)
            if randomize_starts:
                seats = np.random.choice(avail, size=P, replace=False)
            else:
                # Use deterministic perimeter seats but skip any that are dead.
                deterministic_seats = np.array(
                    [i for i, n in enumerate(base.nodes) if n.owner is not None],
                    dtype=np.int64,
                )
                # If a deterministic seat is dead, swap in a non-dead replacement.
                seats = []
                taken_dead = set(int(d) for d in np.where(dead_mask_np[g])[0])
                used = set()
                spare = [i for i in avail.tolist() if i not in used]
                for s_idx in deterministic_seats.tolist():
                    if s_idx in taken_dead:
                        replacement = next(i for i in spare if i not in used and i not in taken_dead)
                        seats.append(replacement)
                        used.add(replacement)
                    else:
                        seats.append(s_idx)
                        used.add(s_idx)
                seats = np.array(seats, dtype=np.int64)
            for p, cell in enumerate(seats[:P]):
                init_owner_np[g, int(cell)] = p
                init_strength_np[g, int(cell)] = seat_base_strength
    else:
        # Deterministic seats (matches the original behaviour).
        init_owner_np[:] = np.array(
            [-1 if n.owner is None else n.owner for n in base.nodes], dtype=np.int32,
        )

    owner = mx.array(init_owner_np)
    strength = mx.array(init_strength_np)
    dead_mask = mx.array(dead_mask_np)
    neighbors = mx.array(neighbors_np)
    live_mask = mx.array(np.ones(G, dtype=np.bool_))

    # Dense flow tensors, refreshed each AI tick.
    flow_src = mx.broadcast_to(mx.arange(N).reshape(1, N), (G, N)).astype(mx.int32)
    flow_dst = mx.zeros((G, N), dtype=mx.int32)
    flow_player = mx.full((G, N), -1, dtype=mx.int32)
    flow_valid = mx.zeros((G, N), dtype=mx.bool_)
    # Passthrough carry (one-tick-lagged friendly inflow on sending cells).
    # Only used by the regen-flow ruleset; transfer-flow step ignores it.
    passthrough = mx.zeros((G, N), dtype=mx.float32)

    # Storage.
    owners_log: list[np.ndarray] = []
    strengths_log: list[np.ndarray] = []
    actions_log: list[np.ndarray] = []
    log_probs_log: list[np.ndarray] = []
    values_log: list[np.ndarray] = []
    rewards_log: list[np.ndarray] = []
    dones_log: list[np.ndarray] = []

    # For recording a single game's frames (game 0). Lower record_stride =
    # finer playback resolution but more Python object allocation per rollout.
    frames: list[GameState] = []
    decision_tick = np.full(G, config.max_ticks, dtype=np.int64)

    # Track cells per seat for delta reward.
    prev_cells = _cells_per_seat(init_owner_np, P)  # (G, P)

    # Initial frame for the recorded game (game 0).
    init_owner_g0 = init_owner_np[0]
    init_strength_g0 = init_strength_np[0]
    frames.append(GameState(
        nodes=tuple(
            GameNode(id=n.id, pos=n.pos,
                     owner=None if int(init_owner_g0[i]) == -1 else int(init_owner_g0[i]),
                     strength=float(init_strength_g0[i]))
            for i, n in enumerate(base.nodes)
        ),
        edges=base.edges,
        flows=tuple(),
        tick=0,
        num_players=P,
    ))

    for t in range(1, config.max_ticks + 1):
        # AI tick: forward, sample, override aggressive, log.
        if t % config.ai_period_ticks == 0:
            if lookahead_k > 0:
                # Search-augmented selection: K candidates, one-step lookahead,
                # pick the action with highest sum-of-seat value at post-state.
                # `logits` is the policy at the *current* (pre-action) state, so
                # PPO's old_log_prob = log of policy prob of the chosen action.
                actions, logits, value, rng_key = lookahead_action_select(
                    model, owner, strength, passthrough, neighbors, dead_mask,
                    live_mask, P, rng_key, K=lookahead_k,
                )
            else:
                # Forward (no gradient needed during rollout).
                logits, value = model(owner, strength, neighbors, P, dead_mask)
                # Sample actions per (G, S, N).
                actions, rng_key = _categorical_sample(logits, rng_key)
            # Single sync: pull everything we need into NumPy at once.
            mx.eval(logits, value, actions, owner, strength)
            owner_np = np.array(owner, copy=False)
            strength_np = np.array(strength, copy=False)
            actions_np = np.array(actions, copy=False)

            # Aggressive override on cells owned by aggressive_seat.
            if aggressive_seat >= 0:
                agg_actions_per_cell = _aggressive_actions_np(
                    owner_np, strength_np, neighbors_np, aggressive_seat
                )  # (G, N)
                # Inject into actions_np at [g, aggressive_seat, cell] for cells the seat owns.
                for g in range(G):
                    mask = owner_np[g] == aggressive_seat
                    actions_np[g, aggressive_seat, mask] = agg_actions_per_cell[g, mask]
                actions = mx.array(actions_np)

            # Log probability of the chosen action per (G, S, cell), summed over cells.
            log_probs_per_cell = _log_probs_at_actions(logits, actions)  # (G, S, N)
            log_probs_per_seat = log_probs_per_cell.sum(axis=-1)  # (G, S)
            mx.eval(log_probs_per_seat)

            # Store.
            owners_log.append(owner_np.copy())
            strengths_log.append(strength_np.copy())
            actions_log.append(actions_np.copy())
            log_probs_log.append(np.array(log_probs_per_seat, copy=False).copy())
            values_log.append(np.array(value, copy=False).copy())
            dones_log.append((~np.array(live_mask, copy=False)).copy())

            # Build flow tensors from these actions.
            flow_src, flow_dst, flow_player, flow_valid = build_flows_from_actions(
                actions, owner, neighbors, dead_mask,
            )

        # Step.
        step_out = step_fn(
            owner, strength, flow_src, flow_dst, flow_player, flow_valid,
            live_mask, P, 0.1, passthrough, neighbors,
        )
        new_owner, new_strength, new_flow_valid, passthrough, wasted_per_cell = step_out
        owner = new_owner
        strength = new_strength
        flow_valid = new_flow_valid
        # Accumulate wasted-flux across the AI period so a single reward eval at
        # AI ticks captures all ticks since the last reward.
        if t % config.ai_period_ticks == 1:
            wasted_acc = wasted_per_cell
        else:
            wasted_acc = wasted_acc + wasted_per_cell

        # Per-tick frame recording (game 0). One game tick = one replay frame
        # so playback can be studied at refresh-rate resolution.
        if t % record_stride == 0:
            mx.eval(owner, strength, flow_valid, passthrough)
            owner_np_now = np.array(owner, copy=False)
            owner_g = owner_np_now[0]
            strength_g = np.array(strength, copy=False)[0]
            fs0 = np.array(flow_src, copy=False)[0]
            fd0 = np.array(flow_dst, copy=False)[0]
            fp0 = np.array(flow_player, copy=False)[0]
            fv0 = np.array(flow_valid, copy=False)[0]
            # Maxed owned cells: emit fanout arrows to all valid neighbors
            # instead of the single action-chosen arrow.
            maxed_mask = (strength_g >= float(MAX_STRENGTH)) & (owner_g >= 0)
            nb_2d = neighbors_np.reshape(-1, 18)
            flow_list: list[Flow] = []
            for k in range(fv0.shape[0]):
                if maxed_mask[k]:
                    own_k = int(owner_g[k])
                    for nb in nb_2d[k]:
                        if nb >= 0:
                            flow_list.append(Flow(src=k, dst=int(nb), player=own_k))
                elif bool(fv0[k]):
                    src_k = int(fs0[k]); dst_k = int(fd0[k]); pl_k = int(fp0[k])
                    if src_k >= 0 and dst_k >= 0 and pl_k >= 0:
                        flow_list.append(Flow(src=src_k, dst=dst_k, player=pl_k))
            game_flows = tuple(flow_list)
            frames.append(GameState(
                nodes=tuple(
                    GameNode(id=n.id, pos=n.pos,
                             owner=None if int(owner_g[i]) == -1 else int(owner_g[i]),
                             strength=float(strength_g[i]))
                    for i, n in enumerate(base.nodes)
                ),
                edges=base.edges,
                flows=game_flows,
                tick=t,
                num_players=P,
            ))
        else:
            owner_np_now = None  # only computed on the AI-tick branch below

        # AI-tick reward: cells delta per seat (only for live games).
        if t % config.ai_period_ticks == 0:
            need_flow_valid = (
                config.engagement_coef != 0.0
                or config.idle_capped_coef != 0.0
                or config.output_boost_coef != 0.0
            )
            need_passthrough = config.output_boost_coef != 0.0
            need_wasted = config.wasted_flux_coef != 0.0
            if owner_np_now is None:
                evals = [owner, strength]
                if need_flow_valid:
                    evals.append(flow_valid)
                if need_passthrough:
                    evals.append(passthrough)
                if need_wasted:
                    evals.append(wasted_acc)
                mx.eval(*evals)
                owner_np_now = np.array(owner, copy=False)
            elif need_flow_valid or need_passthrough or need_wasted:
                extras = []
                if need_flow_valid: extras.append(flow_valid)
                if need_passthrough: extras.append(passthrough)
                if need_wasted: extras.append(wasted_acc)
                mx.eval(*extras)
            live_np = np.array(live_mask, copy=False)
            cells_now = _cells_per_seat(owner_np_now, P)
            delta = (cells_now - prev_cells).astype(np.float32)
            if not config.cell_delta_reward:
                delta[:] = 0.0

            # Dense shaping: reward sending + power output, penalize idle-while-capped.
            # All terms are normalized fractions of cells_owned, so dead seats
            # contribute 0 naturally.
            if (config.engagement_coef != 0.0 or config.idle_capped_coef != 0.0
                    or config.output_boost_coef != 0.0):
                strength_np_now = np.array(strength, copy=False)
                flow_valid_np = np.array(flow_valid, copy=False)
                cells_owned_safe = np.maximum(cells_now, 1).astype(np.float32)
                near_cap = strength_np_now >= config.idle_cap_threshold  # (G, N)
                if config.output_boost_coef != 0.0:
                    passthrough_np = np.array(passthrough, copy=False)
                    cell_regen_np = REGEN_BASE_PER_SEC * (1.0 + REGEN_SLOPE * (strength_np_now - 1.0))
                    output_rate_np = np.minimum(
                        cell_regen_np + passthrough_np, MAX_OUTPUT_PER_SEC,
                    )
                    output_norm_np = output_rate_np / MAX_OUTPUT_PER_SEC  # ∈ [0, 1]
                else:
                    output_norm_np = None
                for s in range(P):
                    seat_mask_np = owner_np_now == s
                    if config.engagement_coef != 0.0:
                        sending = seat_mask_np & flow_valid_np
                        delta[:, s] += (
                            config.engagement_coef
                            * sending.sum(axis=1).astype(np.float32)
                            / cells_owned_safe[:, s]
                        )
                    if config.idle_capped_coef != 0.0:
                        idle_full = seat_mask_np & (~flow_valid_np) & near_cap
                        delta[:, s] -= (
                            config.idle_capped_coef
                            * idle_full.sum(axis=1).astype(np.float32)
                            / cells_owned_safe[:, s]
                        )
                    if output_norm_np is not None:
                        sending = seat_mask_np & flow_valid_np
                        delta[:, s] += (
                            config.output_boost_coef
                            * (output_norm_np * sending).sum(axis=1).astype(np.float32)
                            / cells_owned_safe[:, s]
                        )

            # Wasted-flux penalty: friendly inflow that hit MAX_STRENGTH and
            # got clipped. Accumulated across the AI period above.
            if config.wasted_flux_coef != 0.0:
                wasted_np = np.array(wasted_acc, copy=False)  # (G, N)
                # Normalize by MAX_STRENGTH so the penalty is in 0..N range per game.
                wasted_norm = wasted_np / float(MAX_STRENGTH)
                for s in range(P):
                    seat_mask_np = owner_np_now == s
                    delta[:, s] -= (
                        config.wasted_flux_coef
                        * (wasted_norm * seat_mask_np).sum(axis=1).astype(np.float32)
                        / cells_owned_safe[:, s]
                    )

            # Zero rewards for already-done games.
            delta = delta * live_np.reshape(G, 1).astype(np.float32)
            rewards_log.append(delta.copy())
            prev_cells = cells_now

            # Win check + live-mask update.
            alive = _alive_per_game(owner_np_now, P)
            lm = live_np.copy()
            for g in range(G):
                if not lm[g]:
                    continue
                if alive[g] <= 1:
                    lm[g] = False
                    decision_tick[g] = t
            if not lm.any():
                # Apply terminal bonus to the winner of each game, on top of
                # the last recorded reward step.
                if config.terminal_win_bonus > 0 and rewards_log:
                    final_cells = _cells_per_seat(owner_np_now, P)
                    winners = final_cells.argmax(axis=1)
                    last = rewards_log[-1]
                    for g in range(G):
                        if final_cells[g, winners[g]] > 0:
                            last[g, winners[g]] += config.terminal_win_bonus
                break
            if not np.array_equal(lm, live_np):
                live_mask = mx.array(lm)

    # If we exited without breaking, pad rewards if needed.
    while len(rewards_log) < len(owners_log):
        rewards_log.append(np.zeros((G, P), dtype=np.float32))

    T = len(owners_log)
    rollout = Rollout(
        owner=np.stack(owners_log, axis=0) if T else np.zeros((0, G, N), dtype=np.int32),
        strength=np.stack(strengths_log, axis=0) if T else np.zeros((0, G, N), dtype=np.float32),
        actions=np.stack(actions_log, axis=0) if T else np.zeros((0, G, S, N), dtype=np.int32),
        log_probs=np.stack(log_probs_log, axis=0) if T else np.zeros((0, G, S), dtype=np.float32),
        values=np.stack(values_log, axis=0) if T else np.zeros((0, G, S), dtype=np.float32),
        rewards=np.stack(rewards_log, axis=0) if T else np.zeros((0, G, S), dtype=np.float32),
        dones=np.stack(dones_log, axis=0) if T else np.zeros((0, G), dtype=np.bool_),
        dead_mask=dead_mask_np.copy() if num_dead_cells > 0 else None,
    )

    rollout.advantages, rollout.returns = compute_gae(
        rollout, gamma=config.gamma, lam=config.gae_lambda,
    )

    return rollout, frames, rng_key, dead_mask_np


def compute_gae(rollout: Rollout, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE-λ advantages + bootstrapped returns. Each (g, s) trajectory is
    treated independently."""
    T, G, S = rollout.values.shape
    adv = np.zeros_like(rollout.values)
    last_adv = np.zeros((G, S), dtype=np.float32)
    # Trajectories end when the game becomes done. dones[t, g] = True means the
    # game was already over at step t — treat as terminal.
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
    # Normalize advantages.
    adv_mean, adv_std = adv.mean(), adv.std() + 1e-8
    adv_norm = (adv - adv_mean) / adv_std
    return adv_norm.astype(np.float32), returns.astype(np.float32)


def ppo_update(
    model: GNNActorCritic,
    optimizer: optim.Optimizer,
    rollout: Rollout,
    neighbors: mx.array,
    config: PPOConfig,
) -> dict[str, float]:
    """Run config.update_epochs of minibatch SGD on the PPO loss."""
    T, G, S, N = rollout.actions.shape
    P = config.num_players
    n_steps = T * G
    indices = np.arange(n_steps)

    # Reshape rollout into flat (n_steps, ...).
    flat_owner = rollout.owner.reshape(n_steps, N)
    flat_strength = rollout.strength.reshape(n_steps, N)
    flat_actions = rollout.actions.reshape(n_steps, S, N)
    flat_old_lp = rollout.log_probs.reshape(n_steps, S)
    flat_old_v = rollout.values.reshape(n_steps, S)
    flat_adv = rollout.advantages.reshape(n_steps, S)
    flat_ret = rollout.returns.reshape(n_steps, S)
    flat_dones = rollout.dones.reshape(n_steps)
    # dead_mask is (G, N), constant across T. Tile to match flat_owner shape.
    if rollout.dead_mask is not None:
        flat_dead_mask = np.tile(rollout.dead_mask, (T, 1))   # (n_steps, N)
    else:
        flat_dead_mask = None

    # Mask: skip steps that were already-done (no learning signal).
    learn_mask = ~flat_dones
    # Skip aggressive seat from policy loss (it doesn't learn).
    seat_mask = np.ones(S, dtype=np.float32)
    if config.aggressive_seat >= 0:
        seat_mask[config.aggressive_seat] = 0.0
    # Hoist seat_mask onto the GPU once instead of per-minibatch.
    seat_mask_mx = mx.array(seat_mask).reshape(1, S)
    mx.eval(seat_mask_mx)

    metrics: dict[str, float] = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
        "clip_fraction": 0.0, "ratio_mean": 0.0, "ratio_max": 0.0,
        "n_updates": 0,
    }

    # Metric side-channel: loss_fn appends mx.array refs here; we eval them
    # alongside the gradient outputs so there's no second forward pass.
    metric_buf: list[tuple[mx.array, ...]] = []
    clip_eps = config.clip_eps

    def loss_fn(model, idx):
        owner_mb = mx.array(flat_owner[idx])              # (B, N)
        strength_mb = mx.array(flat_strength[idx])
        actions_mb = mx.array(flat_actions[idx])           # (B, S, N)
        old_lp_mb = mx.array(flat_old_lp[idx])             # (B, S)
        adv_mb = mx.array(flat_adv[idx])                   # (B, S)
        ret_mb = mx.array(flat_ret[idx])                   # (B, S)
        seat_mask_mb = seat_mask_mx
        dead_mb = mx.array(flat_dead_mask[idx]) if flat_dead_mask is not None else None

        logits, value = model(owner_mb, strength_mb, neighbors, P, dead_mb)
        log_softmax = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        new_lp_per_cell = mx.take_along_axis(
            log_softmax, actions_mb[..., None], axis=-1
        ).squeeze(-1)
        new_lp = new_lp_per_cell.sum(axis=-1)

        ratio = mx.exp(new_lp - old_lp_mb)
        surr1 = ratio * adv_mb
        surr2 = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
        policy_loss_per = -mx.minimum(surr1, surr2)
        denom = mx.maximum(seat_mask_mb.sum() * len(idx), 1.0)
        policy_loss = (policy_loss_per * seat_mask_mb).sum() / denom

        value_loss_per = (value - ret_mb) ** 2
        value_loss = (value_loss_per * seat_mask_mb).sum() / denom

        probs = mx.softmax(logits, axis=-1)
        entropy_per_cell = -(probs * log_softmax).sum(axis=-1)
        entropy = (entropy_per_cell.mean(axis=-1) * seat_mask_mb).sum() / denom

        approx_kl = ((old_lp_mb - new_lp) * seat_mask_mb).sum() / denom

        # PPO diagnostics: how often the surrogate ratio gets clipped + ratio scale.
        clipped = (mx.abs(ratio - 1.0) > clip_eps).astype(mx.float32)
        clip_fraction = (clipped * seat_mask_mb).sum() / denom
        ratio_mean = (ratio * seat_mask_mb).sum() / denom
        ratio_max = (ratio * seat_mask_mb).max()

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

    # Norms (one-shot at end of update — cheap).
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


def write_replay(path: Path, frames: list[GameState], radius: int, distance: int,
                 num_players: int, tick_stride: int, metadata: dict) -> None:
    if not frames:
        return
    header = ReplayHeader(
        radius=radius, distance=distance,
        num_players=num_players,
        num_nodes=len(frames[0].nodes),
        tick_stride=tick_stride, dt_per_tick_ms=100,
        metadata=metadata,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        w = ReplayWriter(f, header)
        for fr in frames:
            w.write_frame(fr)
        w.close()
    os.replace(tmp, path)


def _rollout_diagnostics(rollout: Rollout, P: int) -> dict[str, float]:
    """Cheap rollout stats for wandb. All numpy, no MLX evals."""
    values = rollout.values            # (T, G, S)
    returns = rollout.returns          # (T, G, S)
    rewards = rollout.rewards          # (T, G, S)
    actions = rollout.actions          # (T, G, S, N) — last dim is the 19 action choices
    owner = rollout.owner              # (T, G, N) — start-of-tick owners
    dones = rollout.dones              # (T, G)

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
        explained_var = 0.0
        value_mean = value_std = return_mean = return_std = 0.0

    reward_flat = rewards[live].ravel() if live.any() else rewards.ravel()
    reward_step_mean = float(reward_flat.mean()) if reward_flat.size else 0.0
    reward_step_std = float(reward_flat.std()) if reward_flat.size else 0.0
    reward_step_max = float(reward_flat.max()) if reward_flat.size else 0.0
    reward_step_min = float(reward_flat.min()) if reward_flat.size else 0.0

    # Action distribution across all (T, G, S, N).
    a = actions.ravel()
    if a.size:
        action_counts = np.bincount(a, minlength=19).astype(np.float64)
        action_probs = action_counts / action_counts.sum()
        nz = action_probs > 0
        action_entropy = float(-(action_probs[nz] * np.log(action_probs[nz])).sum())
        action_pick_top_frac = float(action_probs.max())
        # Convention used by build_flows_from_actions: index 18 means "no flow"
        # (i.e. the seat chose to do nothing for that cell).
        action_self_frac = float(action_probs[18])
    else:
        action_entropy = action_pick_top_frac = action_self_frac = 0.0

    # End-of-rollout per-seat cell counts (last recorded tick of game 0..G).
    end_owner = owner[-1]            # (G, N)
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
        "value_mean": value_mean,
        "value_std": value_std,
        "return_mean": return_mean,
        "return_std": return_std,
        "reward_step_mean": reward_step_mean,
        "reward_step_std": reward_step_std,
        "reward_step_max": reward_step_max,
        "reward_step_min": reward_step_min,
        "action_entropy": action_entropy,
        "action_pick_top_frac": action_pick_top_frac,
        "action_self_frac": action_self_frac,
        "cells_max_end": float(cells_max.mean()),
        "cells_min_end": float(cells_min.mean()),
        "dominance": float(dominance.mean()),
        "alive_seats_end": float(alive_seats.mean()),
        "neutral_frac_end": float(neutral_frac.mean()),
    }


# Player palette mirrors src/render/scene.ts so wandb images match the browser.
_PLAYER_PALETTE_RGB = np.array([
    [0x4a, 0x90, 0xe2], [0xe2, 0x4a, 0x4a], [0x4a, 0xe2, 0x8a], [0xe2, 0xc4, 0x4a],
    [0xa4, 0x4a, 0xe2], [0xe2, 0x88, 0x4a], [0x4a, 0xe2, 0xe2], [0xe2, 0x4a, 0x88],
    [0x88, 0xe2, 0x4a], [0xe2, 0xe2, 0x4a], [0x4a, 0x88, 0xe2], [0xa4, 0xe2, 0x4a],
], dtype=np.uint8)
_NEUTRAL_RGB = np.array([0x33, 0x33, 0x33], dtype=np.uint8)


def _render_end_state(rollout: Rollout, P: int) -> np.ndarray | None:
    """Stack per-game end-state ownership into a small RGB grid for wandb.

    Lays games out horizontally; each game becomes a row of N coloured cells.
    Not spatially accurate (no hex layout), just a glance at distribution.
    """
    end_owner = rollout.owner[-1]    # (G, N)
    G, N = end_owner.shape
    if G == 0 or N == 0:
        return None
    img = np.empty((G, N, 3), dtype=np.uint8)
    img[:] = _NEUTRAL_RGB
    for g in range(G):
        for s in range(P):
            mask = end_owner[g] == s
            img[g, mask] = _PLAYER_PALETTE_RGB[s % len(_PLAYER_PALETTE_RGB)]
    return img


def append_index(out_dir: Path, entry: dict, cap: int = 50) -> None:
    import json
    index_path = out_dir / "index.json"
    entries = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text())
        except Exception:
            pass
    entries.append(entry)
    entries.sort(key=lambda e: e.get("saved_at", ""), reverse=True)
    kept = entries[:cap]
    for e in entries[cap:]:
        f = e.get("file")
        if f:
            try:
                (out_dir / f).unlink(missing_ok=True)
            except OSError:
                pass
    index_path.write_text(json.dumps(kept, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=9)
    ap.add_argument("--distance", type=int, default=2,
                    help="graph connectivity. 1 = adjacent hex only (6 neighbors); "
                         "2 = 2-hop (18 neighbors, the original wide topology).")
    ap.add_argument("--record-stride", type=int, default=10,
                    help="record every Nth game tick into the replay (game-0 only). "
                         "1 = tick-by-tick (largest replays, finest playback). "
                         "10 = decimated (default, smaller files, snappier playback).")
    ap.add_argument("--num-players", type=int, default=12)
    ap.add_argument("--games-per-rollout", type=int, default=4)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=5000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=0,
                    help="stop after N iterations (0 = forever)")
    ap.add_argument("--aggressive-seat", type=int, default=0)
    ap.add_argument("--no-cell-delta", action="store_true")
    ap.add_argument("--terminal-win-bonus", type=float, default=50.0)
    ap.add_argument("--engagement-coef", type=float, default=0.01,
                    help="dense reward per AI tick: (cells_sending / cells_owned) * coef. "
                         "Set to 0 to disable.")
    ap.add_argument("--idle-capped-coef", type=float, default=0.02,
                    help="dense penalty per AI tick: (idle_near_cap_cells / cells_owned) * coef. "
                         "Set to 0 to disable.")
    ap.add_argument("--idle-cap-threshold", type=float, default=90.0,
                    help="strength >= this counts as 'near cap' for the idle-capped penalty.")
    ap.add_argument("--wasted-flux-coef", type=float, default=0.10,
                    help="penalty for friendly inflow that gets clipped at MAX_STRENGTH "
                         "(regen-flow only). 0 disables.")
    ap.add_argument("--output-boost-coef", type=float, default=0.05,
                    help="dense reward per AI tick: avg (output_rate / MAX_OUTPUT_PER_SEC) "
                         "of sending cells * coef. Regen-flow only — meaningless for transfer.")
    ap.add_argument("--num-dead-cells", type=int, default=0,
                    help="dead cells per game (random per game). Dead cells "
                         "can't be touched — flows to them are dropped.")
    ap.add_argument("--randomize-starts", action="store_true",
                    help="random seat placements per game (default: deterministic "
                         "evenly-spaced perimeter).")
    ap.add_argument("--lookahead-k", type=int, default=0,
                    help="if >0, per AI tick sample K candidate joint actions from the "
                         "policy, simulate one step, pick the one with highest sum-of-seat "
                         "value. Rough stand-in for MCTS; ~K× rollout cost.")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="defaults to checkpoints/ppo/latest.npz for transfer ruleset, "
                         "checkpoints/ppo-regen/latest.npz for regen-flow")
    ap.add_argument("--ruleset", choices=("transfer", "regen-flow"), default="transfer",
                    help="game rules. 'transfer' = original (sender bleeds health). "
                         "'regen-flow' = new rules (sender forfeits regen, symmetric damage).")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="flux")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    args = ap.parse_args()

    if args.checkpoint is None:
        sub = "ppo-regen" if args.ruleset == "regen-flow" else "ppo"
        args.checkpoint = REPO_ROOT / "python" / "checkpoints" / sub / "latest.npz"

    def _step_transfer_adapter(o, s, fs, fd, fp, fv, lm, P, dt, _passthrough, _neighbors=None):
        nw, ns, nfv = step_batched(o, s, fs, fd, fp, fv, lm, P, dt)
        zero_w = mx.zeros(o.shape, dtype=mx.float32)
        return nw, ns, nfv, _passthrough, zero_w

    step_fn = step_batched_regen if args.ruleset == "regen-flow" else _step_transfer_adapter

    config = PPOConfig(
        radius=args.radius, distance=args.distance,
        num_players=args.num_players,
        games_per_rollout=args.games_per_rollout,
        ai_period_ticks=args.ai_period_ticks,
        max_ticks=args.max_ticks,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps, entropy_coef=args.entropy_coef, value_coef=args.value_coef,
        lr=args.lr, update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
        aggressive_seat=args.aggressive_seat,
        cell_delta_reward=not args.no_cell_delta,
        terminal_win_bonus=args.terminal_win_bonus,
        engagement_coef=args.engagement_coef,
        idle_capped_coef=args.idle_capped_coef,
        idle_cap_threshold=args.idle_cap_threshold,
        output_boost_coef=args.output_boost_coef,
        wasted_flux_coef=args.wasted_flux_coef,
    )

    base = make_initial_state(args.radius, args.distance, args.num_players)
    neighbors_np = build_neighbor_table(base)
    neighbors = mx.array(neighbors_np)

    model = GNNActorCritic()
    optimizer = optim.Adam(learning_rate=args.lr)
    mx.eval(model.parameters())

    iteration_offset = 0
    if not args.fresh:
        iteration_offset = load_checkpoint(model, args.checkpoint)
        if iteration_offset > 0:
            print(f"resumed from {args.checkpoint.relative_to(REPO_ROOT)} at iter {iteration_offset}")
    if iteration_offset == 0:
        print("starting from fresh policy")

    rng_key = mx.random.key(args.seed)
    np.random.seed(args.seed ^ 0xDEADBEEF)

    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or f"ppo-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print(f"  wandb run: {wandb_run.url}")

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppo-io")

    print(f"training PPO: radius={args.radius} G={args.games_per_rollout} seats={args.num_players} "
          f"max_ticks={args.max_ticks} aggressive_seat={args.aggressive_seat}")

    iteration = iteration_offset
    try:
        while True:
            if args.iterations > 0 and (iteration - iteration_offset) >= args.iterations:
                break
            iteration += 1

            t0 = time.time()
            rollout, frames, rng_key, dead_mask_np = collect_rollout(
                model, config, neighbors_np, rng_key, step_fn=step_fn,
                record_stride=args.record_stride,
                num_dead_cells=args.num_dead_cells,
                randomize_starts=args.randomize_starts,
                lookahead_k=args.lookahead_k,
            )
            t_rollout = time.time() - t0

            t1 = time.time()
            metrics = ppo_update(model, optimizer, rollout, neighbors, config)
            t_update = time.time() - t1

            mean_reward = float(rollout.rewards.sum(axis=0).mean())
            total_steps = rollout.actions.shape[0]
            roll_diag = _rollout_diagnostics(rollout, config.num_players)
            print(f"iter {iteration:4d}  rollout={t_rollout:.2f}s update={t_update:.2f}s "
                  f"steps={total_steps}  mean_total_R={mean_reward:.2f}  "
                  f"pol_loss={metrics['policy_loss']:.4f} "
                  f"val_loss={metrics['value_loss']:.2f} "
                  f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f} "
                  f"ev={roll_diag['explained_variance']:.2f}")

            if wandb_run is not None:
                log_payload = {
                    "iteration": iteration,
                    "rollout_sec": t_rollout, "update_sec": t_update,
                    "rollout_steps": total_steps,
                    "mean_total_reward": mean_reward,
                    # --- PPO update diagnostics ---
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "approx_kl": metrics["approx_kl"],
                    "clip_fraction": metrics["clip_fraction"],
                    "ratio_mean": metrics["ratio_mean"],
                    "ratio_max": metrics["ratio_max"],
                    "grad_norm": metrics["grad_norm"],
                    "weight_norm": metrics["weight_norm"],
                    # --- value head sanity ---
                    "explained_variance": roll_diag["explained_variance"],
                    "value_mean": roll_diag["value_mean"],
                    "value_std": roll_diag["value_std"],
                    "return_mean": roll_diag["return_mean"],
                    "return_std": roll_diag["return_std"],
                    # --- raw reward signal ---
                    "reward_step_mean": roll_diag["reward_step_mean"],
                    "reward_step_std": roll_diag["reward_step_std"],
                    "reward_step_max": roll_diag["reward_step_max"],
                    "reward_step_min": roll_diag["reward_step_min"],
                    # --- action distribution over the 19 outputs ---
                    "action_entropy": roll_diag["action_entropy"],
                    "action_pick_top_frac": roll_diag["action_pick_top_frac"],
                    "action_self_frac": roll_diag["action_self_frac"],
                    # --- game outcomes per rollout (G games × P seats) ---
                    "cells_max_end": roll_diag["cells_max_end"],
                    "cells_min_end": roll_diag["cells_min_end"],
                    "dominance": roll_diag["dominance"],
                    "alive_seats_end": roll_diag["alive_seats_end"],
                    "neutral_frac_end": roll_diag["neutral_frac_end"],
                }
                # Heavy artifact: end-state ownership image, every 20 iters.
                # Tolerated to fail — wandb.Image needs PIL, which is optional.
                if iteration % 20 == 0:
                    img = _render_end_state(rollout, config.num_players)
                    if img is not None:
                        try:
                            log_payload["end_state"] = wandb.Image(
                                img, caption=f"iter {iteration} game 0 end"
                            )
                        except Exception as e:
                            print(f"  (skipping end_state image: {e})")
                wandb_run.log(log_payload, step=iteration)

            # Replay write (game 0).
            if frames:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                tag = "ppo_regen" if args.ruleset == "regen-flow" else "ppo"
                name = f"train_{tag}_{stamp}_i{iteration:05d}.flxr"
                replay_path = out_dir / name
                # Game 0's dead cells, recorded so the browser can render them
                # distinctly. Stored as a list of cell indices.
                dead_cells_game0 = (
                    [int(i) for i in np.where(dead_mask_np[0])[0]]
                    if args.num_dead_cells > 0 else []
                )
                metadata = {
                    "kind": f"train_{tag}",
                    "model": tag,
                    "ruleset": args.ruleset,
                    "iteration": iteration,
                    "generation": iteration,           # for browser top-bar compatibility
                    "best_fitness": mean_reward,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "dead_cells": dead_cells_game0,
                    "aggressive_seat": args.aggressive_seat,
                }
                def _do_write(p, fr, r, d, np_, ts, m, _it=iteration, _tag=tag, _rs=args.ruleset):
                    try:
                        write_replay(p, fr, r, d, np_, ts, m)
                        append_index(out_dir, {
                            "file": p.name, "saved_at": m["saved_at"],
                            "kind": f"train_{_tag}", "model": _tag,
                            "ruleset": _rs,
                            "iteration": _it,
                            "generation": _it,
                            "radius": r, "distance": d, "num_players": np_,
                        })
                    except Exception as e:
                        import traceback
                        print(f"  REPLAY WRITE FAILED iter {_it}: {type(e).__name__}: {e}")
                        traceback.print_exc()
                write_executor.submit(
                    _do_write,
                    replay_path, frames, args.radius, args.distance, args.num_players,
                    args.record_stride, metadata,
                )

            if iteration % 5 == 0:
                # Checkpoint.
                write_executor.submit(save_checkpoint, model, args.checkpoint, iteration)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        write_executor.shutdown(wait=True)
        save_checkpoint(model, args.checkpoint, iteration)
        print(f"checkpoint saved → {args.checkpoint.relative_to(REPO_ROOT)}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
