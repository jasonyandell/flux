"""Scan public/replays/*.flxr, filter to long positive-fitness games, write greatest-hits.json.

Walks every replay binary, reads the FLXR header + metadata JSON, and emits an
ordered list of "greatest hits" entries for the browser to cycle through.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY_DIR = REPO_ROOT / "public" / "replays"
DEFAULT_OUT = DEFAULT_REPLAY_DIR / "greatest-hits.json"

MAGIC = b"FLXR"
HEADER_FIXED = 28  # 4 (magic) + 20 (packed fields) + 4 (metadata_len)


def read_header(path: Path) -> dict | None:
    try:
        with path.open("rb") as f:
            head = f.read(HEADER_FIXED)
            if len(head) < HEADER_FIXED or head[:4] != MAGIC:
                return None
            (_version, _r1, radius, distance, num_players, _r2,
             num_nodes, tick_stride, dt_per_tick_ms, num_frames) = struct.unpack(
                "<BBhhBBIHHI", head[4:24]
            )
            (metadata_len,) = struct.unpack("<I", head[24:28])
            meta_bytes = f.read(metadata_len)
            metadata = json.loads(meta_bytes.decode("utf-8")) if metadata_len else {}
        duration_sec = num_frames * tick_stride * dt_per_tick_ms / 1000.0
        return {
            "file": path.name,
            "saved_at": metadata.get("saved_at"),
            "kind": metadata.get("kind"),
            "model": metadata.get("model"),
            "iteration": metadata.get("iteration"),
            "generation": metadata.get("generation"),
            "radius": radius,
            "num_players": num_players,
            "num_frames": num_frames,
            "duration_sec": round(duration_sec, 2),
            "best_fitness": metadata.get("best_fitness"),
        }
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-frames", type=int, default=200,
                    help="skip replays with fewer recorded frames")
    ap.add_argument("--min-fitness", type=float, default=0.0,
                    help="skip replays with fitness at or below this")
    ap.add_argument("--top", type=int, default=30,
                    help="keep at most this many entries")
    args = ap.parse_args()

    entries: list[dict] = []
    for path in sorted(args.replay_dir.glob("*.flxr")):
        e = read_header(path)
        if e is None:
            continue
        fit = e.get("best_fitness")
        if fit is None or not isinstance(fit, (int, float)):
            continue
        if fit <= args.min_fitness:
            continue
        if e["num_frames"] < args.min_frames:
            continue
        entries.append(e)

    # Sort: longest first as the primary key (the user wants long games), then
    # fitness as the tiebreaker so within a length tier we get the best play.
    entries.sort(key=lambda e: (e["num_frames"], e["best_fitness"]), reverse=True)
    entries = entries[: args.top]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {len(entries)} entries → {args.out.relative_to(REPO_ROOT)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
