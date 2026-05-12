"""Binary replay format for flux games.

A replay is a sampled snapshot of game state across ticks, designed to be
played back by the browser without re-simulating. Geometry (node positions,
edges) is NOT stored — it's deterministic from (radius, distance, numPlayers)
via make_initial_state, so the player rebuilds it locally.

Layout (little-endian):
  Header (28 bytes fixed + variable metadata):
    0   4   magic "FLXR" (0x46 0x4C 0x58 0x52)
    4   1   version = 1
    5   1   reserved
    6   2   radius (i16)
    8   2   distance (i16)
    10  1   num_players (u8)
    11  1   reserved
    12  4   num_nodes (u32)
    16  2   tick_stride (u16)        ticks between recorded frames
    18  2   dt_per_tick_ms (u16)     hint for playback timing
    20  4   num_frames (u32)
    24  4   metadata_len (u32)
    28  M   metadata bytes (UTF-8 JSON)

  Then num_frames frames, each:
    owner[num_nodes]    i8   (-1 = neutral)
    strength[num_nodes] u8   (round(strength * 255 / MAX_STRENGTH))
    flow_count          u16
    flows[flow_count]   each 5 bytes: src u16, dst u16, player u8

Frame size = 2*N + 2 + 5*F.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .state import GameState, MAX_STRENGTH

MAGIC = b"FLXR"
VERSION = 1


@dataclass
class ReplayHeader:
    radius: int
    distance: int
    num_players: int
    num_nodes: int
    tick_stride: int
    dt_per_tick_ms: int
    metadata: dict


class ReplayWriter:
    """Streams a replay to an open binary file. Patches num_frames on close."""

    def __init__(self, f: BinaryIO, header: ReplayHeader):
        self._f = f
        self._frame_count = 0
        self._num_nodes = header.num_nodes
        metadata_bytes = json.dumps(header.metadata, separators=(",", ":")).encode("utf-8")
        # Write header with num_frames=0; patched on close.
        f.write(MAGIC)
        f.write(struct.pack(
            "<BBhhBBIHHI",
            VERSION,
            0,  # reserved
            header.radius,
            header.distance,
            header.num_players,
            0,  # reserved
            header.num_nodes,
            header.tick_stride,
            header.dt_per_tick_ms,
            0,  # num_frames placeholder
        ))
        f.write(struct.pack("<I", len(metadata_bytes)))
        f.write(metadata_bytes)
        # Position of num_frames field within the header (after magic 4 + 16 fixed):
        # magic(4) + version(1) + reserved(1) + radius(2) + distance(2)
        #   + num_players(1) + reserved(1) + num_nodes(4) + tick_stride(2)
        #   + dt_per_tick_ms(2) = 20. num_frames lives at offset 20.
        self._num_frames_offset = 20

    def write_frame(self, state: GameState) -> None:
        n = len(state.nodes)
        assert n == self._num_nodes, f"node count mismatch: {n} vs {self._num_nodes}"
        owners = bytes(
            (0xFF if node.owner is None else node.owner) & 0xFF
            for node in state.nodes
        )
        # Quantize strength to u8: round(strength * 255 / MAX_STRENGTH), clamp.
        scale = 255.0 / MAX_STRENGTH
        strengths = bytes(
            max(0, min(255, int(node.strength * scale + 0.5)))
            for node in state.nodes
        )
        self._f.write(owners)
        self._f.write(strengths)
        flows = state.flows
        self._f.write(struct.pack("<H", len(flows)))
        for flow in flows:
            self._f.write(struct.pack("<HHB", flow.src, flow.dst, flow.player))
        self._frame_count += 1

    def close(self) -> None:
        # Patch num_frames into the header.
        self._f.seek(self._num_frames_offset)
        self._f.write(struct.pack("<I", self._frame_count))
        self._f.seek(0, 2)  # back to end
        self._f.flush()


def write_replay(
    path: Path,
    header: ReplayHeader,
    frames: list[GameState],
) -> None:
    """Convenience: write a whole replay in one call."""
    with open(path, "wb") as f:
        w = ReplayWriter(f, header)
        for s in frames:
            w.write_frame(s)
        w.close()
