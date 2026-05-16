"""Lightning solver — public surface for all `lightning_*` modes.

The implementation is now fully vectorized in `solver_vec`. This module
exists to keep stable imports for `lightning_solver_actions` and the
`compute_potential` helper used by training scripts.

Modes (selected via `mode=` kwarg):
  max               original max-aggregator (tree-shaped flow)
  sum               uniform Bellman value-iteration; current default winner
  sum_pw            edge-pressure-weighted Bellman
  loop              structural curl (no potential field)
  vortex            CW curl (loop with curl_dir=-1)
  attn              two-head: gradient attack + even-k loop, BFS-distance blend
  attn_release      attn with build-release loop gate (0.7)
  attn_slam         attn with build-release loop gate (0.95)
  flood             set every valid slot
  random            random per-cell action
  chase             counter-attack on inbound threat
  sum_wave          sum + 60% MAX gate ("pulse")
  max_wave          max + 60% MAX gate
  wave_keep_attack  sum_wave but frontier never throttles
  pulse             global tick-synchronized charge/fire
  pulse_stagger     even/odd cells out of phase

See `wiki/topics/v2-overnight-research.md` for the matched-pair ranking.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .solver_vec import compute_potential, lightning_actions
from .state import State


def lightning_solver_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
    gamma: float = 0.85,
    weak_bonus: float = 1.0,
    expand_bonus: float = 0.6,
    defense_bonus: float = 0.0,
    fanout_eps: float = 0.05,
    mode: str = "max",
    curl_dir: int = 1,
    **mode_kwargs,
) -> np.ndarray:
    """Vectorized dispatch — see `solver_vec.lightning_actions` for modes."""
    return lightning_actions(
        state, seat, rng,
        gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
        defense_bonus=defense_bonus, fanout_eps=fanout_eps,
        mode=mode, curl_dir=curl_dir, **mode_kwargs,
    )


__all__ = ["lightning_solver_actions", "compute_potential"]
