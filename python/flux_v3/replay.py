"""FLXR v2 binary replay format.

Layout (little-endian) — same fixed header as v1 (magic + version + radius
+ ...), with version=2. Per-frame flow record is **6 bytes** instead of 5:

  src u16, dst u16, player u8, pressure_q u8

`pressure_q` is the directed edge pressure quantized to MAX_EDGE → 0..255.

Geometry is NOT stored — it's derived from (radius, num_players) by the
v2 board builder (`make_board`) on the player side.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .state import MAX_EDGE, MAX_STRENGTH, K

MAGIC = b"FLXR"
VERSION = 2


@dataclass
class ReplayHeader:
    radius: int
    num_players: int
    num_nodes: int
    tick_stride: int
    dt_per_tick_ms: int
    metadata: dict


@dataclass
class Frame:
    """One recorded frame of game 0 state.

    owner          (N,)        int32   -2 dead / -1 neutral / 0..P-1 seat
    strength       (N,)        float32
    flows          list of (src, dst, player, pressure)
    """
    owner: np.ndarray
    strength: np.ndarray
    flows: list[tuple[int, int, int, float]]


def state_to_frame(state) -> Frame:
    """Project a v2 State into a Frame.

    Each active outflow slot becomes one flow record with its current
    edge_pressure value.
    """
    flows: list[tuple[int, int, int, float]] = []
    N = state.N
    for c in range(N):
        owner_c = int(state.owner[c])
        if owner_c < 0:
            continue
        for k in range(K):
            if not bool(state.outflow[c, k]):
                continue
            d = int(state.neighbors[c, k])
            if d < 0:
                continue
            pressure = float(state.edge_pressure[c, k])
            flows.append((c, d, owner_c, pressure))
    return Frame(
        owner=np.asarray(state.owner, dtype=np.int32).copy(),
        strength=np.asarray(state.strength, dtype=np.float32).copy(),
        flows=flows,
    )


class ReplayWriter:
    def __init__(self, f: BinaryIO, header: ReplayHeader) -> None:
        self._f = f
        self._frame_count = 0
        self._num_nodes = header.num_nodes
        # Note: distance is fixed to 1 in v2 (K=6 direct neighbors); we still
        # encode it in the same i16 slot so the header is byte-compatible with
        # v1 readers that ignore the version byte.
        metadata_bytes = json.dumps(header.metadata, separators=(",", ":")).encode("utf-8")
        f.write(MAGIC)
        f.write(struct.pack(
            "<BBhhBBIHHI",
            VERSION,
            0,                          # reserved
            header.radius,
            1,                          # "distance" — always 1 in v2
            header.num_players,
            0,                          # reserved
            header.num_nodes,
            header.tick_stride,
            header.dt_per_tick_ms,
            0,                          # num_frames placeholder
        ))
        f.write(struct.pack("<I", len(metadata_bytes)))
        f.write(metadata_bytes)
        self._num_frames_offset = 20

    def write_frame(self, frame: Frame) -> None:
        n = frame.owner.shape[0]
        assert n == self._num_nodes, f"frame nodes {n} != header {self._num_nodes}"
        # owner u8 with two's-complement (Python doesn't sign-encode; use & 0xFF).
        owner_bytes = bytes((int(o) & 0xFF) for o in frame.owner)
        # strength quantized to u8.
        scale = 255.0 / MAX_STRENGTH
        strength_bytes = bytes(
            max(0, min(255, int(s * scale + 0.5))) for s in frame.strength
        )
        self._f.write(owner_bytes)
        self._f.write(strength_bytes)
        flows = frame.flows
        self._f.write(struct.pack("<H", len(flows)))
        edge_scale = 255.0 / MAX_EDGE
        for src, dst, player, pressure in flows:
            pq = max(0, min(255, int(pressure * edge_scale + 0.5)))
            self._f.write(struct.pack("<HHBB", src, dst, player, pq))
        self._frame_count += 1

    def close(self) -> None:
        self._f.seek(self._num_frames_offset)
        self._f.write(struct.pack("<I", self._frame_count))
        self._f.seek(0, 2)
        self._f.flush()


def write_replay(
    path: Path, header: ReplayHeader, frames: list[Frame]
) -> None:
    """Convenience wrapper that writes to a temp path then renames."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        w = ReplayWriter(f, header)
        for fr in frames:
            w.write_frame(fr)
        w.close()
    os.replace(tmp, path)


def append_index(out_dir: Path, entry: dict, cap: int = 50) -> None:
    """Maintain a JSON index of recent replays, newest-first, capped at `cap`."""
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
