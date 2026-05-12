"""Run one flux game on MLX (single game, multiple seats from one genome) and
write a replay file. Uses mlx_step + mlx_genome end-to-end.

Usage:
    uv run python scripts/play_one_mlx.py
    uv run python scripts/play_one_mlx.py --radius 18 --num-players 12 --ticks 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from flux import build_neighbor_table, make_initial_state  # noqa: E402
from flux.mlx_genome import nn_infer_all_cells, random_genome  # noqa: E402
from flux.mlx_step import step_mlx  # noqa: E402
from flux.replay import ReplayHeader, ReplayWriter  # noqa: E402
from flux.state import (  # noqa: E402
    Flow,
    GameNode,
    GameState,
    MAX_STRENGTH,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "public" / "replays"


def mlx_to_python_state(
    base: GameState,
    owner_np: np.ndarray,
    strength_np: np.ndarray,
    flows: list[tuple[int, int, int]],
) -> GameState:
    """Build a GameState for replay-writing from MLX-derived arrays."""
    new_nodes = tuple(
        GameNode(
            id=n.id,
            pos=n.pos,
            owner=None if int(owner_np[i]) == -1 else int(owner_np[i]),
            strength=float(strength_np[i]),
        )
        for i, n in enumerate(base.nodes)
    )
    new_flows = tuple(Flow(src=s, dst=d, player=p) for (s, d, p) in flows)
    return GameState(
        nodes=new_nodes,
        edges=base.edges,
        flows=new_flows,
        tick=base.tick,
        num_players=base.num_players,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=9)
    ap.add_argument("--distance", type=int, default=2)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--ai-period-ticks", type=int, default=1,
                    help="run NN every N ticks (1 = every tick)")
    ap.add_argument("--tick-stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--genome-std", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (out_dir / f"mlx_{stamp}_seed{args.seed}.flxr")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    base = make_initial_state(args.radius, args.distance, args.num_players)
    neighbors_np = build_neighbor_table(base)
    N = len(base.nodes)
    P = args.num_players

    # Initial state → MLX tensors.
    owner = mx.array(
        np.array([-1 if n.owner is None else n.owner for n in base.nodes], dtype=np.int32)
    )
    strength = mx.array(
        np.array([n.strength for n in base.nodes], dtype=np.float32)
    )
    neighbors = mx.array(neighbors_np)

    # One genome drives all seats (each seat is a different player index, so the
    # NN's friend/enemy features will produce per-seat behaviour even with shared
    # weights). The evolution loop will instead hand out one genome per seat.
    genome = random_genome(mx.random.key(args.seed), std=args.genome_std)
    mx.eval(owner, strength, neighbors, genome)

    metadata = {
        "kind": "mlx_play_one",
        "seed": args.seed,
        "genome_std": args.genome_std,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "note": "single-game MLX step + NN, one shared genome across seats",
    }
    header = ReplayHeader(
        radius=args.radius,
        distance=args.distance,
        num_players=args.num_players,
        num_nodes=N,
        tick_stride=args.tick_stride,
        dt_per_tick_ms=100,
        metadata=metadata,
    )

    # Flow representation: list of (src, dst, player) tuples in Python.
    flows: list[tuple[int, int, int]] = []

    def frame_state(tick: int) -> GameState:
        owner_np = np.array(owner, copy=False)
        strength_np = np.array(strength, copy=False)
        s = mlx_to_python_state(base, owner_np, strength_np, flows)
        # Replay writer doesn't read tick, but keep it semantically correct.
        return GameState(
            nodes=s.nodes,
            edges=s.edges,
            flows=s.flows,
            tick=tick,
            num_players=s.num_players,
        )

    t0 = time.time()
    with open(tmp_path, "wb") as f:
        w = ReplayWriter(f, header)
        w.write_frame(frame_state(0))

        for t in range(1, args.ticks + 1):
            # 1) NN per player → desired_flow per player-owned cell.
            if t % args.ai_period_ticks == 0:
                # `desired` is a dict { (player, cell): dst }
                # We rebuild it fresh each AI tick.
                desired: dict[tuple[int, int], int] = {}
                for p in range(P):
                    actions = nn_infer_all_cells(genome, owner, strength, neighbors, p)
                    mx.eval(actions)
                    actions_np = np.array(actions, copy=False)
                    owner_np = np.array(owner, copy=False)
                    for cell in range(N):
                        if int(owner_np[cell]) != p:
                            continue
                        a = int(actions_np[cell])
                        if a < 0 or a >= 18:
                            continue
                        dst = int(neighbors_np[cell * 18 + a])
                        if dst < 0:
                            continue
                        desired[(p, cell)] = dst

                # 2) Reconcile flows: keep flows that match desired (and erase
                # them from `desired`); drop flows that don't.
                kept: list[tuple[int, int, int]] = []
                for (src, dst, player) in flows:
                    key = (player, src)
                    d = desired.get(key)
                    if d is not None and d == dst:
                        kept.append((src, dst, player))
                        del desired[key]
                    # else: drop (acts as a toggle-cancel)
                # 3) Add new desired flows.
                for (player, src), dst in desired.items():
                    kept.append((src, dst, player))
                flows = kept

            # 4) Step in MLX.
            if flows:
                flow_src = mx.array(np.array([s for (s, _, _) in flows], dtype=np.int32))
                flow_dst = mx.array(np.array([d for (_, d, _) in flows], dtype=np.int32))
                flow_player = mx.array(np.array([p for (_, _, p) in flows], dtype=np.int32))
            else:
                flow_src = mx.array(np.zeros(0, dtype=np.int32))
                flow_dst = mx.array(np.zeros(0, dtype=np.int32))
                flow_player = mx.array(np.zeros(0, dtype=np.int32))

            new_owner, new_strength, alive = step_mlx(
                owner, strength, flow_src, flow_dst, flow_player, P, 0.1
            )
            mx.eval(new_owner, new_strength, alive)
            owner = new_owner
            strength = new_strength
            if flows:
                alive_np = np.array(alive, copy=False)
                flows = [fl for fl, keep in zip(flows, alive_np.tolist()) if keep]

            if t % args.tick_stride == 0:
                w.write_frame(frame_state(t))
        w.close()
    os.replace(tmp_path, out_path)

    sz = out_path.stat().st_size
    dt = time.time() - t0
    print(f"wrote {out_path.relative_to(REPO_ROOT)}  ticks={args.ticks}  "
          f"frames={1 + args.ticks // args.tick_stride}  "
          f"size={sz} bytes  elapsed={dt:.2f}s  "
          f"ticks/sec={args.ticks/dt:.1f}")

    # Index update.
    index_path = out_dir / "index.json"
    entries: list[dict] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text())
            if not isinstance(entries, list):
                entries = []
        except json.JSONDecodeError:
            entries = []
    entries.append({
        "file": out_path.name,
        "saved_at": metadata["saved_at"],
        "kind": metadata["kind"],
        "seed": args.seed,
        "ticks": args.ticks,
        "radius": args.radius,
        "num_players": args.num_players,
    })
    entries.sort(key=lambda e: e.get("saved_at", ""), reverse=True)
    entries = entries[:50]
    index_path.write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    main()
