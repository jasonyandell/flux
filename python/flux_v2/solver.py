"""Old-school algorithmic solver for flux v2 — BFS frontier-relay heuristic.

Public surface kept stable; the implementation is now the vectorized one in
`solver_vec`. The per-cell Python loop that used to live here was 30%+ of v2
single-game wall time on R=20 boards.

Per-cell ideal outflow set for seat `s` at cell `c`:

  attack/expand   slot points at enemy or neutral (not dead, not off-grid)
  relay           slot points at friendly `d` with dist_to_frontier[d] < dist[c]
                  AND `d` is not a dead-end MAX-strength sink
  excluded        everything else

Priority per AI tick: SET missing attack > SET missing relay > CLEAR stale > NOOP.

Board precondition (see `wiki/decisions/v2-board-connectivity.md`): every
non-DEAD cell must be reachable from every other through non-DEAD neighbors.
Isolated live pockets never reach the global frontier through relays.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .solver_vec import bfs_actions
from .state import State


def solver_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Vectorized BFS solver — see `solver_vec.bfs_actions`."""
    return bfs_actions(state, seat, rng)


def _pick(candidates: np.ndarray, rng: Optional[np.random.Generator]) -> int:
    """Legacy helper retained for external sweep scripts."""
    if rng is None:
        return int(candidates[0])
    return int(candidates[rng.integers(candidates.size)])
