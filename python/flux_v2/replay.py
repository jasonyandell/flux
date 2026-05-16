"""FLXR v3 binary replay format.

Layout (little-endian):

  magic        4 bytes   "FLXR"
  version      u8        = 3
  reserved     u8
  header_len   u32       length of JSON metadata blob
  header_json  bytes     UTF-8 JSON: radius, num_players, num_nodes,
                                    tick_stride, dt_per_tick_ms, num_frames,
                                    max_strength, max_edge, metadata
  frames_gz    bytes     gzip-compressed frame stream (until EOF)

Frame stream (uncompressed; concatenated, fixed-size per game):
  owners            N bytes int8    -2 dead, -1 neutral, 0..P-1 seat
  strengths         N bytes uint8   quantized 0..255 over [0, max_strength]
  outflow_bits      ceil(N*K/8)     bit i: cell (i//K), slot (i%K)
  pressure_bytes    popcount bytes  uint8 quantized over [0, max_edge],
                                    in the same iteration order as outflow_bits

Geometry is NOT stored — derived from (radius, num_players) on the player
side via `make_board`.

Reducer, action space, geometry are unchanged from v2; only the wire format
is. Old v1/v2 readers don't speak v3. Per project direction (2026-05-15):
historical replays are disposable and quickly regeneratable, so we don't
carry a back-compat path.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .state import MAX_EDGE, MAX_STRENGTH, K

MAGIC = b"FLXR"
VERSION = 3


@dataclass
class ReplayHeader:
    radius: int
    num_players: int
    num_nodes: int
    tick_stride: int
    dt_per_tick_ms: int
    metadata: dict
    max_strength: float = MAX_STRENGTH
    max_edge: float = MAX_EDGE


@dataclass
class Frame:
    """One recorded frame of game state.

    owner          (N,)        int8    -2 dead / -1 neutral / 0..P-1 seat
    strength       (N,)        float32 in [0, max_strength]
    outflow        (N, K)      bool
    edge_pressure  (N, K)      float32 in [0, max_edge]
    """
    owner: np.ndarray
    strength: np.ndarray
    outflow: np.ndarray
    edge_pressure: np.ndarray


def state_to_frame(state) -> Frame:
    """Project a v2 State into a Frame."""
    return Frame(
        owner=np.asarray(state.owner, dtype=np.int8).copy(),
        strength=np.asarray(state.strength, dtype=np.float32).copy(),
        outflow=np.asarray(state.outflow, dtype=np.bool_).copy(),
        edge_pressure=np.asarray(state.edge_pressure, dtype=np.float32).copy(),
    )


def _encode_frame(
    frame: Frame, max_strength: float, max_edge: float,
) -> bytes:
    """Pack one frame to dense bytes (pre-compression)."""
    owners_bytes = np.asarray(frame.owner, dtype=np.int8).tobytes()
    s_scale = 255.0 / max_strength
    s_q = np.clip(np.rint(frame.strength * s_scale), 0, 255).astype(np.uint8)
    strengths_bytes = s_q.tobytes()
    outflow_flat = np.asarray(frame.outflow, dtype=np.bool_).reshape(-1)
    outflow_packed = np.packbits(outflow_flat, bitorder="big").tobytes()
    e_scale = 255.0 / max_edge
    ep_flat = np.asarray(frame.edge_pressure, dtype=np.float32).reshape(-1)
    ep_active = ep_flat[outflow_flat]
    ep_q = np.clip(np.rint(ep_active * e_scale), 0, 255).astype(np.uint8)
    pressures_bytes = ep_q.tobytes()
    return owners_bytes + strengths_bytes + outflow_packed + pressures_bytes


class ReplayWriter:
    """Streaming writer: header buffered until close along with frame stream.

    Header carries the final frame count, so we accumulate frames in memory
    (dense bytes — cheap) and write everything at close. Memory footprint:
    ~3.5KB × num_frames for R=20 (≈ 13 MB for an 18 k-tick stride-5 run),
    well within process memory.
    """
    def __init__(self, f: BinaryIO, header: ReplayHeader) -> None:
        self._f = f
        self._header = header
        self._frames: list[bytes] = []

    def write_frame(self, frame: Frame) -> None:
        n = frame.owner.shape[0]
        assert n == self._header.num_nodes, (
            f"frame nodes {n} != header {self._header.num_nodes}"
        )
        self._frames.append(_encode_frame(
            frame, self._header.max_strength, self._header.max_edge,
        ))

    def close(self) -> None:
        hdr = {
            "radius": int(self._header.radius),
            "num_players": int(self._header.num_players),
            "num_nodes": int(self._header.num_nodes),
            "tick_stride": int(self._header.tick_stride),
            "dt_per_tick_ms": int(self._header.dt_per_tick_ms),
            "num_frames": len(self._frames),
            "max_strength": float(self._header.max_strength),
            "max_edge": float(self._header.max_edge),
            "metadata": self._header.metadata,
        }
        header_json = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        self._f.write(MAGIC)
        self._f.write(struct.pack("<BB", VERSION, 0))
        self._f.write(struct.pack("<I", len(header_json)))
        self._f.write(header_json)
        raw = b"".join(self._frames)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
            gz.write(raw)
        self._f.write(buf.getvalue())
        self._f.flush()


def write_replay(
    path: Path, header: ReplayHeader, frames: list[Frame],
) -> None:
    """Convenience wrapper: writes to a temp path then renames."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        w = ReplayWriter(f, header)
        for fr in frames:
            w.write_frame(fr)
        w.close()
    os.replace(tmp, path)


def append_index(out_dir: Path, entry: dict, cap: int = 50) -> None:
    """Maintain a JSON index of recent replays, newest-first, capped at `cap`.

    Also append-once to `events.jsonl` so the viewer can show new-arrival
    badges without diffing index snapshots. The log is the source of truth
    for "what's happened"; index.json is the latest snapshot.
    """
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

    events_path = out_dir / "events.jsonl"
    event = {
        "ts": entry.get("saved_at", ""),
        "event": "replay_added",
        "file": entry.get("file"),
        "iteration": entry.get("iteration"),
        "generation": entry.get("generation"),
        "radius": entry.get("radius"),
        "num_players": entry.get("num_players"),
        "kind": entry.get("kind"),
    }
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
