"""flux v2 state — pressure-game core.

A node has an owner and a scalar strength. A *cell* has six outgoing slots
(direct hex neighbors). Each directed edge carries a persistent
`edge_pressure` that's rewritten each tick from the source cell's overflow.

The reducer is pure: `tick(state) → state'` produces a fresh State, never
mutates in place. Actions only mutate `outflow_intent`, and only at AI ticks
via `apply_actions`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Hex topology — fixed at K=6 direct neighbors per cell.
K: int = 6

# Per-PRD constants. "Big-bag-of-pressure" rules (2026-05-15):
#   - Caps scaled 10x so reservoirs can grow large enough to be
#     interesting; "MAWSSIVE shots" deliverable in one tick.
#   - REGEN scaled 10x to preserve fill timescale (200 ticks 0→MAX).
#   - CAPTURE_STRENGTH no longer used — captures now keep the surplus
#     pressure that overcame the defender (see step.py capture line).
MAX_STRENGTH: float = 1000.0
MAX_EDGE: float = 1000.0
CAPTURE_STRENGTH: float = 50.0         # legacy; new rule uses surplus pressure
REGEN_BASE_PER_TICK: float = 5.0       # baseline cell regen per game tick
REGEN_SLOPE: float = 0.0               # linear scaling; 0 = flat regen

# Waste = "any regen you didn't send." If a cell has outflows set, its regen
# is "sent" (whether or not the per-edge cap clips some of it — the cap is a
# system limit, not the policy's fault). Only no-spill counts as waste: a
# cell with no outflows and overflowing regen is true waste because the
# regen wasn't directed anywhere.
WASTE_WEIGHT_NO_SPILL: float = 1.0
WASTE_WEIGHT_CAP_BOUND: float = 0.0
# Pressure that lands on a friendly cell which is BOTH at MAX strength AND has
# zero active outflows — a pure sink with no relay. Pass-through MAX cells
# (with outflows) are combo relays, not waste. See the v2-three-term-reward
# decision doc for the source/destination split.
WASTE_WEIGHT_DEST_TERMINATED: float = 0.3

# Transit credit is the positive twin of destination-terminated waste. It
# attributes pressure to a source cell when that pressure lands on a friendly
# relay instead of a sink. Strict mode means "relay" requires a MAX-strength
# destination with active outflows; this targets the combo/back-line pattern
# without paying ordinary fill traffic.
TRANSIT_CREDIT_STRICT: bool = True

# Action space: 13 actions per cell per AI tick.
ACTION_SET_BASE: int = 0
ACTION_CLEAR_BASE: int = K
ACTION_NOOP: int = 2 * K
NUM_ACTIONS: int = 2 * K + 1           # 13

# Owner sentinels.
NEUTRAL: int = -1
DEAD: int = -2

# Hex axial direction vectors, indexed by slot k.
# Slot k and slot (k+3)%6 are opposite directions (the "back-edge").
HEX_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (+1,  0),
    (+1, -1),
    ( 0, -1),
    (-1,  0),
    (-1, +1),
    ( 0, +1),
)

OPPOSITE_SLOT: np.ndarray = np.array(
    [(k + 3) % K for k in range(K)], dtype=np.int32,
)


@dataclass
class State:
    """v2 game state. Arrays are owned by this dataclass; the reducer copies
    them when producing a new state so callers can't accidentally mutate the
    previous tick's tensors.

    `outflow` is the policy's persistent decision (mutated at AI ticks).
    `edge_pressure` is what's flowing through each directed slot this tick
    (mutated every tick from the cell's overflow).
    """
    N: int
    pos: np.ndarray            # (N, 2) float32
    coord: np.ndarray          # (N, 2) int32 — axial (q, r)
    neighbors: np.ndarray      # (N, 6) int32 — slot k → cell id (-1 if off-grid)
    owner: np.ndarray          # (N,) int32 — NEUTRAL / DEAD / 0..P-1
    strength: np.ndarray       # (N,) float32
    outflow: np.ndarray        # (N, 6) bool
    edge_pressure: np.ndarray  # (N, 6) float32
    tick: int
    num_players: int
    waste_total: float = 0.0   # cumulative waste across the game (diagnostic)


def regen(strength: np.ndarray) -> np.ndarray:
    """Per-tick regen rate per cell. Linear knob is open in the PRD."""
    return REGEN_BASE_PER_TICK * (1.0 + REGEN_SLOPE * (strength - 1.0))


def copy_state(s: State) -> State:
    return State(
        N=s.N,
        pos=s.pos,                           # immutable layout, safe to share
        coord=s.coord,
        neighbors=s.neighbors,
        owner=s.owner.copy(),
        strength=s.strength.copy(),
        outflow=s.outflow.copy(),
        edge_pressure=s.edge_pressure.copy(),
        tick=s.tick,
        num_players=s.num_players,
        waste_total=s.waste_total,
    )
