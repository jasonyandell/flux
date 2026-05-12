"""Run one flux game with a random NN controller and write a replay file.

Proof-of-pipeline: uses the existing NumPy step + NN, no MLX yet. The point
is to exercise the Python → public/replays/ → browser path end-to-end.

Usage:
    uv run python scripts/play_one.py
    uv run python scripts/play_one.py --out ../public/replays/test.bin --ticks 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a script from python/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flux import (  # noqa: E402
    ai_think,
    apply_action,
    build_neighbor_table,
    make_initial_state,
    mulberry32,
    random_genome,
    step,
)
from flux.replay import ReplayHeader, ReplayWriter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "public" / "replays"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=9)
    ap.add_argument("--distance", type=int, default=2)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--tick-stride", type=int, default=5,
                    help="record a frame every N ticks")
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    ap.add_argument("--genome-std", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or (out_dir / f"play_{stamp}_seed{args.seed}.flxr")

    state = make_initial_state(args.radius, args.distance, args.num_players)
    rng = mulberry32(args.seed)
    genome = random_genome(rng, std=args.genome_std)
    neighbors = build_neighbor_table(state)

    metadata = {
        "kind": "play_one",
        "seed": args.seed,
        "genome_std": args.genome_std,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "note": "single-game smoke test with random NN, NumPy step",
    }
    header = ReplayHeader(
        radius=args.radius,
        distance=args.distance,
        num_players=args.num_players,
        num_nodes=len(state.nodes),
        tick_stride=args.tick_stride,
        dt_per_tick_ms=100,
        metadata=metadata,
    )

    t0 = time.time()
    # Write to a .tmp file then atomic-rename so the browser never sees partial.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        w = ReplayWriter(f, header)
        # Record initial state as frame 0.
        w.write_frame(state)
        for t in range(1, args.ticks + 1):
            if t % args.ai_period_ticks == 0:
                for p in range(state.num_players):
                    for a in ai_think(state, p, genome, neighbors):
                        state = apply_action(state, a)
            state = step(state, 0.1)
            if t % args.tick_stride == 0:
                w.write_frame(state)
        w.close()
    os.replace(tmp_path, out_path)

    sz = out_path.stat().st_size
    dt = time.time() - t0
    print(f"wrote {out_path.relative_to(REPO_ROOT)}  ticks={args.ticks}  "
          f"frames={1 + args.ticks // args.tick_stride}  "
          f"size={sz} bytes  elapsed={dt:.2f}s")

    # Update the index.
    update_index(out_dir, out_path, metadata, args)


def update_index(out_dir: Path, replay_path: Path, metadata: dict, args) -> None:
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
        "file": replay_path.name,
        "saved_at": metadata["saved_at"],
        "kind": metadata["kind"],
        "seed": args.seed,
        "ticks": args.ticks,
        "radius": args.radius,
        "num_players": args.num_players,
    })
    # Newest first; cap at 50.
    entries.sort(key=lambda e: e.get("saved_at", ""), reverse=True)
    entries = entries[:50]
    index_path.write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    main()
