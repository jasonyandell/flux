"""Auxiliary pretraining smoke/train loop for the v2 edge-aware policy.

Samples synthetic v2 board states, derives edge category/channel targets from
the same helper used by the edge model, then trains the auxiliary heads. This
does not alter PPO rollouts; it is a cheap representation check before RL.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx                                                # noqa: E402
import mlx.nn as nn                                                  # noqa: E402
import mlx.optimizers as optim                                       # noqa: E402
import numpy as np                                                   # noqa: E402

from flux_v2.graph import make_board, random_seat_and_dead            # noqa: E402
from flux_v2.mlx_step import tick_batched                             # noqa: E402
from flux_v2.ppo import (                                             # noqa: E402
    EDGE_CHANNEL_NAMES,
    EDGE_NUM_CHANNELS,
    EdgeAwareActorCritic,
    build_edge_auxiliary_targets_np,
)
from flux_v2.state import DEAD, K, MAX_STRENGTH, NEUTRAL              # noqa: E402


@dataclass
class EdgeAuxBatch:
    owner: mx.array
    strength: mx.array
    outflow: mx.array
    edge_pressure: mx.array
    labels: mx.array
    channels: mx.array
    mask: mx.array


def _sample_state_batch(
    *,
    radius: int,
    num_players: int,
    batch_size: int,
    num_dead_cells: int,
    territory_prob: float,
    outflow_prob: float,
    warmup_ticks: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = make_board(radius, num_players)
    N = base.N
    neighbors = base.neighbors
    owner = np.full((batch_size, N), NEUTRAL, dtype=np.int32)
    strength = np.zeros((batch_size, N), dtype=np.float32)

    for g in range(batch_size):
        seats, dead = random_seat_and_dead(
            N,
            num_players,
            num_dead_cells,
            rng,
            neighbors=neighbors,
            min_seat_dist=2,
            coord=base.coord,
        )
        owner_g = np.full(N, NEUTRAL, dtype=np.int32)
        strength_g = rng.uniform(5.0, MAX_STRENGTH, size=N).astype(np.float32)
        if len(dead) > 0:
            owner_g[dead] = DEAD
            strength_g[dead] = 0.0

        live = owner_g != DEAD
        claimed = live & (rng.random(N) < territory_prob)
        owner_g[claimed] = rng.integers(0, num_players, size=int(claimed.sum()))
        for p, cell in enumerate(seats):
            owner_g[int(cell)] = p
            strength_g[int(cell)] = 30.0

        owner[g] = owner_g
        strength[g] = strength_g

    outflow = np.zeros((batch_size, N, K), dtype=np.bool_)
    for g in range(batch_size):
        for c in range(N):
            if owner[g, c] < 0:
                continue
            for k in range(K):
                d = int(neighbors[c, k])
                if d >= 0 and owner[g, d] != DEAD and rng.random() < outflow_prob:
                    outflow[g, c, k] = True

    edge_pressure = np.zeros((batch_size, N, K), dtype=np.float32)
    if warmup_ticks > 0:
        o = mx.array(owner)
        st = mx.array(strength)
        of = mx.array(outflow)
        ep = mx.array(edge_pressure)
        alive = mx.array(np.ones(batch_size, dtype=np.bool_))
        nb = mx.array(neighbors)
        for _ in range(warmup_ticks):
            o, st, of, ep, _, _ = tick_batched(o, st, of, ep, alive, num_players, nb)
        mx.eval(o, st, of, ep)
        owner = np.array(o, copy=False)
        strength = np.array(st, copy=False)
        outflow = np.array(of, copy=False)
        edge_pressure = np.array(ep, copy=False)

    return owner, strength, outflow, edge_pressure, neighbors


def sample_batch(args: argparse.Namespace, rng: np.random.Generator) -> EdgeAuxBatch:
    owner, strength, outflow, edge_pressure, neighbors = _sample_state_batch(
        radius=args.radius,
        num_players=args.num_players,
        batch_size=args.batch_size,
        num_dead_cells=args.num_dead_cells,
        territory_prob=args.territory_prob,
        outflow_prob=args.outflow_prob,
        warmup_ticks=args.warmup_ticks,
        rng=rng,
    )
    labels, channels, valid_mask = build_edge_auxiliary_targets_np(
        owner, strength, outflow, edge_pressure, neighbors, args.num_players,
    )
    if args.include_blocked:
        mask = np.ones_like(valid_mask, dtype=np.float32)
    else:
        mask = valid_mask.astype(np.float32)
    return EdgeAuxBatch(
        owner=mx.array(owner),
        strength=mx.array(strength),
        outflow=mx.array(outflow),
        edge_pressure=mx.array(edge_pressure),
        labels=mx.array(labels),
        channels=mx.array(channels),
        mask=mx.array(mask),
    )


def edge_aux_loss(
    model: EdgeAwareActorCritic,
    batch: EdgeAuxBatch,
    neighbors: mx.array,
    num_players: int,
    channel_coef: float,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    type_logits, channel_pred = model.edge_auxiliary(
        batch.owner,
        batch.strength,
        batch.outflow,
        batch.edge_pressure,
        neighbors,
        num_players,
    )
    logp = type_logits - mx.logsumexp(type_logits, axis=-1, keepdims=True)
    chosen = mx.take_along_axis(logp, batch.labels[..., None], axis=-1).squeeze(-1)
    denom = mx.maximum(batch.mask.sum(), 1.0)
    type_loss = -(chosen * batch.mask).sum() / denom
    channel_loss = (
        ((channel_pred - batch.channels) ** 2) * batch.mask[..., None]
    ).sum() / (denom * float(EDGE_NUM_CHANNELS))
    return type_loss + channel_coef * channel_loss, (type_loss, channel_loss)


def evaluate(
    model: EdgeAwareActorCritic,
    batch: EdgeAuxBatch,
    neighbors: mx.array,
    num_players: int,
    channel_coef: float,
) -> dict[str, float]:
    loss, (type_loss, channel_loss) = edge_aux_loss(
        model, batch, neighbors, num_players, channel_coef,
    )
    type_logits, channel_pred = model.edge_auxiliary(
        batch.owner,
        batch.strength,
        batch.outflow,
        batch.edge_pressure,
        neighbors,
        num_players,
    )
    mx.eval(loss, type_loss, channel_loss, type_logits, channel_pred)
    pred = np.array(type_logits).argmax(axis=-1)
    labels = np.array(batch.labels)
    mask = np.array(batch.mask).astype(bool)
    accuracy = float((pred[mask] == labels[mask]).mean()) if mask.any() else 0.0
    channel_mse = float(channel_loss)
    channel_mean = np.array(channel_pred)[mask].mean(axis=0) if mask.any() else np.zeros(EDGE_NUM_CHANNELS)
    out = {
        "loss": float(loss),
        "type_loss": float(type_loss),
        "channel_loss": channel_mse,
        "type_accuracy": accuracy,
    }
    for name, value in zip(EDGE_CHANNEL_NAMES, channel_mean):
        out[f"pred_{name}_mean"] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-dead-cells", type=int, default=20)
    ap.add_argument("--territory-prob", type=float, default=0.45)
    ap.add_argument("--outflow-prob", type=float, default=0.20)
    ap.add_argument("--warmup-ticks", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--channel-coef", type=float, default=1.0)
    ap.add_argument("--include-blocked", action="store_true")
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = make_board(args.radius, args.num_players)
    neighbors = mx.array(base.neighbors)
    model = EdgeAwareActorCritic()
    optimizer = optim.Adam(learning_rate=args.lr)
    mx.eval(model.parameters())

    def loss_fn(model, batch):
        loss, aux = edge_aux_loss(
            model, batch, neighbors, args.num_players, args.channel_coef,
        )
        return loss

    grad_fn = nn.value_and_grad(model, loss_fn)
    for step in range(1, args.steps + 1):
        batch = sample_batch(args, rng)
        loss, grads = grad_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, batch, neighbors, args.num_players, args.channel_coef,
            )
            print(
                f"step {step:04d} loss={metrics['loss']:.4f} "
                f"type={metrics['type_loss']:.4f} channel={metrics['channel_loss']:.4f} "
                f"acc={metrics['type_accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
